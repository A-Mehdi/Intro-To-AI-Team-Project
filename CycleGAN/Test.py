import os
import cv2
from models.Test_Config import Face2Cartoon


if __name__ == '__main__':
    Input_path = './test_images/'
    Save_path = './save_images/'
    filename_list = os.listdir(Input_path)
    for i in filename_list:
        save_name = i.split('.')[0] + '.png'
        img = cv2.cvtColor(cv2.imread(f'./test_images/{i}'), cv2.COLOR_BGR2RGB)
        c2p = Face2Cartoon()
        cartoon = c2p.inference(img)
        if cartoon is not None:
            cv2.imwrite(Save_path + save_name, cartoon)