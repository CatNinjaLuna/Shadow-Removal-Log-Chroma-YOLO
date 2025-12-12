import os
import shutil
import argparse

def rename_tif_files(source_dir, target_dir):
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Target directory created: {target_dir}")
    
    processed_count = 0
    skipped_count = 0
    

    for filename in os.listdir(source_dir):

        if filename.lower().endswith('.tif'):

            if "_log_chroma" in filename:

                new_filename = filename.replace("_log_chroma", "")            

                source_path = os.path.join(source_dir, filename)
                target_path = os.path.join(target_dir, new_filename)
                
                try:

                    shutil.copy2(source_path, target_path)
                    print(f"Processed: {filename} -> {new_filename}")
                    processed_count += 1
                except Exception as e:
                    print(f"Error when processing {filename}: {str(e)}")
            else:
                print(f"skip: {filename} (does not include 'log_chroma')")
                skipped_count += 1
    
    print("\nProcess completed")
    print(f"Numbers of files processed: {processed_count}")
    print(f"Skipping {skipped_count} files")

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eliminate "_log_chroma" in file namings')
    parser.add_argument('--source_dir', default=r'/projects/SuperResolutionData/carolinali-shadowRemoval/log_chroma_output_test_1', help='source file path')
    parser.add_argument('--target_dir', default=r'/projects/SuperResolutionData/carolinali-shadowRemoval/output_yolo/log_chroma', help='target file path')
    
    args = parser.parse_args()
    
    rename_tif_files(args.source_dir, args.target_dir)
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eliminate "_log_chroma" in file namings')
    parser.add_argument('--source_dir', default=r'/projects/SuperResolutionData/carolinali-shadowRemoval/my_training_data/log_chroma_output', help='source file path')
    parser.add_argument('--target_dir', default=r'/projects/SuperResolutionData/carolinali-shadowRemoval/output_yolo_all/log_chroma', help='target file path')
    
    args = parser.parse_args()
    
    rename_tif_files(args.source_dir, args.target_dir)