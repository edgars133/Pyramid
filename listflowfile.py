import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.pfm'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    image = [img for img in classes if img.find('RGB_cleanpass') > -1]
    disp  = [dsp for dsp in classes if dsp.find('disparity') > -1]

    print(len(disp))
    monkaa_path = filepath #+ [x for x in image if '/content/drive/MyDrive/Group70Pyramid/Sampler/Monkaa/' in x][0]
    monkaa_disp = filepath #+ [x for x in disp if '/content/drive/MyDrive/Group70Pyramid/Sampler/Monkaa/' in x][0]

    monkaa_dir  = os.listdir(monkaa_path)

    all_left_img=[]
    all_right_img=[]
    all_left_disp = []
    test_left_img=[]
    test_right_img=[]
    test_left_disp = []

    count=0
    for dd in monkaa_dir:
      #print(dd)
      count=count+1
      for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
        if count==1 and is_image_file(monkaa_path+'/'+dd+'/left/'+im):
          all_left_img.append(monkaa_path+'/'+dd+'/left/'+im)
        if count>1:
          all_left_disp.append(monkaa_disp+'/'+dd+'/left/'+im.split(".")[0]+'.pfm')
#'/'+dd+'/left/'+
      for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
        if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
          all_right_img.append(monkaa_path+'/'+dd+'/right/'+im)
    print((all_left_img))
    print((all_right_img))
    print((all_left_disp))
    '''
    flying_path = filepath + [x for x in image if x == 'RGB_cleanpass'][0]
    flying_disp = filepath + [x for x in disp if x == 'RGB_cleanpass'][0]
    flying_dir = flying_path+'/TRAIN/'
    subdir = ['A','B','C']

    for ss in subdir:
      flying = os.listdir(flying_dir+ss)

      for ff in flying:
        imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
        for im in imm_l:
          if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
            all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
          
          all_left_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

          if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
            all_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

    flying_dir = flying_path+'/TEST/'

    subdir = ['A','B','C']

    for ss in subdir:
      flying = os.listdir(flying_dir+ss)

      for ff in flying:
        imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
        for im in imm_l:
          if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
            test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
          
          test_left_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

          if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
            test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)


    driving_dir = filepath + [x for x in image if 'Driving' in x][0] + '/'
    driving_disp = filepath + [x for x in disp if 'Driving' in x][0]

    subdir1 = ['35mm_focallength','15mm_focallength']
    subdir2 = ['scene_backwards','scene_forwards']
    subdir3 = ['fast','slow']

    for i in subdir1:
      for j in subdir2:
        for k in subdir3:
            imm_l = os.listdir(driving_dir+i+'/'+j+'/'+k+'/left/')    
            for im in imm_l:
              if is_image_file(driving_dir+i+'/'+j+'/'+k+'/left/'+im):
                all_left_img.append(driving_dir+i+'/'+j+'/'+k+'/left/'+im)

              all_left_disp.append(driving_disp+'/'+i+'/'+j+'/'+k+'/left/'+im.split(".")[0]+'.pfm')

              if is_image_file(driving_dir+i+'/'+j+'/'+k+'/right/'+im):
                all_right_img.append(driving_dir+i+'/'+j+'/'+k+'/right/'+im)
    '''

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp


