import os
import ffmpeg

def preprocess(video, output, fps=12, width=720, height=1280):
    try:
        stream = ffmpeg.input(video)
        stream = stream.filter('fps', fps=fps, round='up')
        
        out = f"train_processed/{output}"
        
        stream = ffmpeg.output(stream, out)
        
        ffmpeg.run(stream, quiet=False)
    except ffmpeg.Error as e:
        error_message = e.stderr.decode('utf-8') if e.stderr else "Unknown FFmpeg error"
        print("FFmpeg Error:", error_message)
        raise

if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs("train_processed", exist_ok=True)

    # Get list of processed and raw files
    processed = sorted(os.listdir("train_processed"))
    raw = sorted(os.listdir("pushup_raw"))

    # Determine the starting count
    if len(processed) == 0:
        count = 0
    else:
        # Extract count from the last processed file
        count = int(processed[-1][:3])
        count += 1

    print(f"Starting count: {count}")

    for i in raw:
        # Skip non-video files
        if not i.endswith('.mp4'):
            print(f"Skipping non-video file: {i}")
            continue

        # File path for the raw video
        file = f"pushup_raw/{i}"
        
        # Extract the form name from the raw filename
        form_name = '_'.join(i.split('_')[:-1])  # Everything before the last underscore
        
        # Format the new filename with leading count and form name
        leading_count = str(count).zfill(3)
        name = f"{leading_count}_{form_name}.mp4"

        try:
            # Preprocess the video
            preprocess(file, name)
            print(f"Processed: {name}")
            
            # Remove the original raw file after successful processing
            os.remove(file)
            count += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")
