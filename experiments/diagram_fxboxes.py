import os
import shutil

selected_img = '150458'
print(selected_img)

count_img = 0
count_selectedfx = 0

copy_path = 'diagram_images'

if not os.path.isdir(copy_path):
    os.mkdir(copy_path)
    
for root, dirs, files in os.walk('/home/ece/Desktop/phd_2022/gaze_code/segment/fx2SAM_cropped'):

    for f in files:

        if '.png' in f:
            count_img += 1

            _, ppn, img_id, fx_id = f.split('.')[0].split('_')

            if img_id == selected_img:
                image_path = os.path.join(root, f)
                shutil.copy2(image_path, copy_path)
                count_selectedfx += 1
                
print(count_img) 
print(count_selectedfx)

