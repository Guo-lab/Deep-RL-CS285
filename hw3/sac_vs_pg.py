import os
import glob
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record


def rename_scalar_in_event_file(input_file, output_file, old_tag, new_tag):
    """
    Rename a scalar tag in a TensorBoard event file.

    Args:
        input_file: Path to input event file
        output_file: Path to output event file
        old_tag: Old scalar tag name (e.g., 'Eval_AverageReturn')
        new_tag: New scalar tag name (e.g., 'eval_return')
    """
    writer = tf_record.TFRecordWriter(output_file)

    for event in summary_iterator(input_file):
        # Check if this event has summary data
        if event.summary:
            for value in event.summary.value:
                # Rename the tag if it matches
                if value.tag == old_tag:
                    value.tag = new_tag

        # Write the modified event
        writer.write(event.SerializeToString())

    writer.close()


def process_all_event_files(data_dir, old_tag="Eval_AverageReturn", new_tag="eval_return"):
    """
    Process all TensorBoard event files in subdirectories.

    Args:
        data_dir: Root data directory (e.g., 'data')
        old_tag: Old scalar tag name
        new_tag: New scalar tag name
    """
    # Find all event files
    pattern = os.path.join(data_dir, "**", "events.out.tfevents.*")
    event_files = glob.glob(pattern, recursive=True)

    print(f"Found {len(event_files)} event files")

    for event_file in event_files:
        print(f"Processing: {event_file}")

        # Create backup with .old extension
        backup_file = event_file + ".old"
        temp_file = event_file + ".tmp"

        try:
            # Rename scalar and write to temp file
            rename_scalar_in_event_file(event_file, temp_file, old_tag, new_tag)

            # Backup original file
            os.rename(event_file, backup_file)

            # Replace original with modified version
            os.rename(temp_file, event_file)

            print(f"  ✓ Renamed '{old_tag}' to '{new_tag}'")
            print(f"  Backup saved as: {backup_file}")

        except Exception as e:
            print(f"  ✗ Error processing {event_file}: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == "__main__":
    # Process all event files in data directory
    process_all_event_files("data/q2_pg_humanoid_Humanoid-v4_31-12-2025_11-43-59")
    process_all_event_files("data/q2_pg_humanoid_Humanoid-v4_31-12-2025_22-59-29")

    print("\nDone! Original files backed up with .old extension")
