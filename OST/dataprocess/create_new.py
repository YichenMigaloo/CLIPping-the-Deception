
import os
import cv2


from util.face_align import align_face  # face_align.py
from util.faceswap import swap_faces  # faceswap.py
from util.face_blend import blend_faces  # face_blend.py
from util.color_transfer import color_transfer  



source_image_dir = "/content/CLIPping-the-Deception/deepfake_eval/faceswap/images/val/n01440764"
target_image_dir = "/content/CLIPping-the-Deception/deepfake_eval/faceswap/images/val/n01443537"
output_dir = "/content/generated"



def process_image_pair(source_image_path, target_image_path, output_path):
    # Step 1: Load source and target images
    source_img = cv2.imread(source_image_path)
    target_img = cv2.imread(target_image_path)

    if source_img is None or target_img is None:
        print(f"Error reading {source_image_path} or {target_image_path}")
        return

    # Step 2: Face Alignment
    aligned_face, aligned_target = align_face(source_img, target_img)

    if aligned_face is None or aligned_target is None:
        print(f"Face alignment failed for {source_image_path} or {target_image_path}")
        return

    # Step 3: Face Swap
    swapped_face = swap_faces(aligned_face, aligned_target)

    if swapped_face is None:
        print(f"Face swapping failed for {source_image_path} and {target_image_path}")
        return

    # Step 4: Blend Faces (for seamless merging)
    blended_image = blend_faces(swapped_face, aligned_target)

    # Step 5: Color Transfer (to match color tones)
    final_image = color_transfer(blended_image, aligned_target)

    # Save the generated fake image
    cv2.imwrite(output_path, final_image)

    print(f"Generated fake image saved to {output_path}")

# Loop through all pairs of source and target images
for source_img_name in os.listdir(source_image_dir):
    for target_img_name in os.listdir(target_image_dir):
        source_img_path = os.path.join(source_image_dir, source_img_name)
        target_img_path = os.path.join(target_image_dir, target_img_name)

        # Generate unique output filename
        output_path = os.path.join(output_dir, f"fake_{source_img_name.split('.')[0]}_to_{target_img_name}")

        # Process the image pair and generate fake image
        process_image_pair(source_img_path, target_img_path, output_path)
