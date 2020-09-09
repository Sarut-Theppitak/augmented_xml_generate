# 2020/09/04
# Created by Sarut Theppitak
# causetion. If you run this program in the IDE, you wull not beable to stop the process
# if you run by command line, type ctrl+c will kill all the process and please close the teminal

import os
import time
import pandas as pd
import numpy as np
from lxml import etree
import xml.etree.ElementTree as ET
import argparse
from PIL import Image, ImageEnhance
from multiprocessing import Pool
from multiprocessing import cpu_count
from pathlib import Path
import imgaug.augmenters as iaa  # https://github.com/aleju/imgaug
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 20)



class img_augmentor:
    def __init__(self,ori_img_folder,ori_csv, aug_type, out_folder,procs):
        self.ori_csv = ori_csv
        self.ori_img_folder = ori_img_folder
        last_folder = ori_img_folder.split('\\')[-1]
        self.aug_type = aug_type
        self.out_folder = out_folder
        self.img_out_folder= os.path.join(out_folder, last_folder + '_' + aug_type)
        self.xml_out_folder= self.img_out_folder + '_xml'
        self.csv_out_path= self.img_out_folder + '.csv'
        self.procs = procs if procs > 0 else cpu_count()
        self.procIDs = list(range(0, procs))
        fields = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class', 'x', 'y']
        self.df = pd.read_csv(self.ori_csv, usecols=fields)
        self.img_width = self.df.loc[0,'x']
        self.img_height = self.df.loc[0,'y']
        self.img_names = pd.unique(self.df['filename'])
        numImagesPerProc = len(self.img_names) / float(self.procs)
        self.numImagesPerProc = int(np.ceil(numImagesPerProc))   

    def df_to_xml(self, df):
        #check if the path is correct if not create the folder
        Path(self.xml_out_folder).mkdir(parents=True, exist_ok=True)
        
        for img_name in pd.unique(df['filename']):
        
            sub_df = df[df['filename'] == img_name]
            
            #the hight and width information will be extract from the first entry of its file_name
            height = df['y'].iloc[0]
            width = df['x'].iloc[0]
            #specify the num of chanel of you image 
            depth = 1
        
            annotation = ET.Element('annotation')
            ET.SubElement(annotation, 'folder').text = self.img_out_folder.split('\\')[-1]
            ET.SubElement(annotation, 'filename').text = img_name
            ET.SubElement(annotation, 'path').text = os.path.join(self.img_out_folder,img_name)
            source = ET.SubElement(annotation, 'source')
            ET.SubElement(source, 'database').text = 'Unknown'
            size = ET.SubElement(annotation, 'size')
            ET.SubElement(size, 'width').text = str(width)
            ET.SubElement(size, 'height').text = str(height)
            ET.SubElement(size, 'depth').text = str(depth)
            ET.SubElement(annotation, 'segmented').text = '0'
            
            obs = {}
            bbox_obs = {}
            for i in range(len(sub_df)):
                
                obs['ob'+ str(i)] = ET.SubElement(annotation, 'object')
                ET.SubElement(obs['ob'+ str(i)], 'name').text = sub_df['class'].iloc[i]
                ET.SubElement(obs['ob'+ str(i)], 'pose').text = 'Unspecified'
                ET.SubElement(obs['ob'+ str(i)], 'truncated').text = '0'
                ET.SubElement(obs['ob'+ str(i)], 'difficult').text = '0'
                bbox_obs['ob'+ str(i)] = ET.SubElement(obs['ob'+ str(i)], 'bndbox')
                ET.SubElement(bbox_obs['ob'+ str(i)], 'xmin').text = str(sub_df['xmin'].iloc[i])
                ET.SubElement(bbox_obs['ob'+ str(i)], 'ymin').text = str(sub_df['ymin'].iloc[i])
                ET.SubElement(bbox_obs['ob'+ str(i)], 'xmax').text = str(sub_df['xmax'].iloc[i])
                ET.SubElement(bbox_obs['ob'+ str(i)], 'ymax').text = str(sub_df['ymax'].iloc[i])
            
            
            img_name_xml = img_name.replace('.png','.xml')
            final_xml_path = os.path.join(self.xml_out_folder,img_name_xml)
            
            tree = ET.ElementTree(annotation)
            tree.write(final_xml_path, encoding='utf8')
    
    def df_to_csv(self, df):
        #check if the path is correct if not create the folder
        Path(self.out_folder).mkdir(parents=True, exist_ok=True)
        df.to_csv(self.csv_out_path, index = False)
    
    def flipVer(self,im):
        processed_im = im.transpose(Image.FLIP_TOP_BOTTOM)
        return processed_im
    
    def flipHor(self,im):
        processed_im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return processed_im
    
    def enhanced(self,im):
        processed_im = ImageEnhance.Contrast(im).enhance(1.3)
        return processed_im  
    
    def rotate180(self,im):
        im = np.array(im)
        seq = iaa.Sequential([iaa.Rot90(2)])
        processed_im = seq(image=im)
        processed_im = Image.fromarray(processed_im)
        return processed_im
    
    def random(self,im,image_name):
        im = np.array(im)
        df_part = self.df[self.df['filename'] == image_name].copy()
        BoundingBoxes= [BoundingBox(x1=row['xmin'], y1=row['ymin'], x2=row['xmax'], y2=row['ymax'], label =row['class'] ) 
                        for index, row in df_part.iterrows()]
        bbs = BoundingBoxesOnImage(BoundingBoxes, shape=im.shape)
        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontal flips
            iaa.Crop(percent=(0, 0.1)), # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ],random_order=True) # apply augmenters in random order
            
        processed_im, bbs_aug = seq(image=im, bounding_boxes=bbs)
        #eliminate outside bounding box
        bbs_aug_out = bbs_aug.remove_out_of_image().clip_out_of_image() 
        
        all_lists = []
        for b in bbs_aug_out:
            list_append = [image_name,int(round(b.x1)),int(round(b.y1)),int(round(b.x2)),int(round(b.y2)),b.label,processed_im.shape[1],processed_im.shape[0]]
            all_lists.append(list_append)
        df_box = pd.DataFrame(all_lists,columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class', 'x', 'y'])
        processed_im = Image.fromarray(processed_im)
        #df_box can have len() = 0
        return processed_im,df_box
    
    def img_augmentation(self,payload):
        print("[INFO] starting process {}".format(payload["id"]))
        method_to_use = getattr(self, self.aug_type)
        ##########################################
        if self.aug_type == 'random':
            df_boxes = []
            for image_name in payload["chuck"]:
                image_path = os.path.join(self.ori_img_folder,image_name)
                save_name =  image_name[:-4] + '_' + self.aug_type + '.png'
                save_path = os.path.join(self.img_out_folder,save_name)
                im = Image.open(image_path)
                processed_im , df_box = method_to_use(im,image_name)
                processed_im.save(save_path,quality=95)
                df_boxes.append(df_box)
            #cancat all dfboxes together
            df = pd.concat(df_boxes,ignore_index= True)
            return df
        
        else:    
            for image_name in payload["chuck"]:
                image_path = os.path.join(self.ori_img_folder,image_name)
                save_name =  image_name[:-4] + '_' + self.aug_type + '.png'
                save_path = os.path.join(self.img_out_folder,save_name)
                im = Image.open(image_path)
                processed_im = method_to_use(im)
                processed_im.save(save_path,quality=95)
            
    def change_df(self,df):
        #change name
        df['filename'] = df['filename'].str[:-4] + '_' + self.aug_type + '.png'
        #change position
        if self.aug_type == 'flipVer':
            df['ymin_new'] = df['y'] - df['ymax']   #height-ymax
            df['ymax_new'] = df['y'] - df['ymin']   #height-ymin
            df['ymin'] = df['ymin_new'] 
            df['ymax'] = df['ymax_new']
            df = df.drop(columns=['ymin_new', 'ymax_new'])
            
        elif self.aug_type == 'flipHor':
            df['xmin_new'] = df['x'] - df['xmax']   #width-xmax
            df['xmax_new'] = df['x']  - df['xmin']  #width-xmax
            df['xmin'] = df['xmin_new'] 
            df['xmax'] = df['xmax_new']
            df = df.drop(columns=['xmin_new', 'xmax_new'])
        elif self.aug_type == 'enhanced':
            pass
        elif self.aug_type == 'rotate180':
            BoundingBoxes= [BoundingBox(x1=row['xmin'], y1=row['ymin'], x2=row['xmax'], y2=row['ymax'], label =row['class'] ) 
                           for index, row in df.iterrows()]
            bbs = BoundingBoxesOnImage(BoundingBoxes, shape=(self.img_height,self.img_width))
            seq = iaa.Sequential([iaa.Rot90(2)])
            bbs_aug = seq(bounding_boxes=bbs)
            df[['xmin','ymin', 'xmax', 'ymax']] = np.rint(bbs_aug.to_xyxy_array()).astype(int)
        else:
            pass
        
        return df
    
    @staticmethod
    def chunk(l, n):
    	# loop over the list in n-sized chunks
    	for i in range(0, len(l), n):
    		# yield the current n-sized chunk to the calling function
    		yield l[i: i + n]        
        
    def augmentor_main(self):
        chunks = list(img_augmentor.chunk(self.img_names, self.numImagesPerProc))
        payloads = []
        
        #check if the path is correct if not create the folder
        Path(self.img_out_folder).mkdir(parents=True, exist_ok=True)
        
        for (i, chuck) in enumerate(chunks):
            # construct a dictionary of data for the payload, then add it
		    # to the payloads list
            payload = {
    			"id": i,
    			"chuck": chuck
    		}
            payloads.append(payload)
            
        # construct and launch the processing pool
        start_time = time.time()
        print("[INFO] launching pool using {} processes...".format(self.procs))
        pool = Pool(processes=self.procs)
        results = pool.map(self.img_augmentation, payloads)
        # close the pool and wait for all processes to finish
        pool.close()
        pool.join()
                
        print("[INFO] multiprocessing of image augmentation, {} type, is completed".format(self.aug_type))
        print("--- %s seconds ---" % (time.time() - start_time))
        
        #Do CSV process
        start_time = time.time()
        print("[INFO] now creating csv for {} type.....".format(self.aug_type))
        ########################################################
        if isinstance(results[0], pd.DataFrame):
            #concat df from each processes 
            df_all = pd.concat(results,ignore_index= True)
            self.df = df_all
            self.df['filename'] = self.df['filename'].str[:-4] + '_' + self.aug_type + '.png'
            ##########################
            ## need to get rid of the bouding box that is out side the image?
        else:
            self.df = self.change_df(self.df)
            
        #create csv file from df here   
        self.df_to_csv(self.df)    
        
        print("[INFO] csv is created")
        print("--- %s seconds ---" % (time.time() - start_time))
        
        #Do XML process
        start_time = time.time()
        print("[INFO] now creating XML files for {} type.....".format(self.aug_type))
        self.df_to_xml(self.df)
        print("[INFO] XML files are created")
        print("--- %s seconds ---" % (time.time() - start_time))
        
             
if __name__ == '__main__': 
#    parser = argparse.ArgumentParser(description= 'Create xml files from CSV file')
#    parser.add_argument('--ori_img_folder', type= str,required = True, help = '# Specify your original image folder')
#    parser.add_argument('--ori_csv', type= str,required = True, help = '# Specify your csv path with .csv extention')
#    parser.add_argument('--aug_type', type= str,required = True, help = '# Specify type of your augmentation')
#    parser.add_argument('--out_folder', type= str,required = True, help = '# Specify your output folder. 
#                         The augmented img folder,augmented xml folder,and augmented csv file will be create in thsi folder ')
#    parser.add_argument('--procs', type=int, default=-1,help="# of processes to spin up")
#    args = parser.parse_args()
#    augment_object = img_augmentor(args.ori_img_folder,args.ori_csv,args.aug_type,
#                                   args.out_folder,args.procs) 
#    augment_object.augmentor_main()    
    ori_img_folder = 'C:\\Users\\3978\\Desktop\\Faster-Rcnn\\inceptionV2_ais_s1_20200826\\data\\sample_img'
    ori_csv = 'C:\\Users\\3978\\Desktop\\Faster-Rcnn\\inceptionV2_ais_s1_20200826\\data\\sample.csv'
    aug_type = 'rotate180'
    out_folder = 'C:\\Users\\3978\\Desktop\\Faster-Rcnn\\inceptionV2_ais_s1_20200826\\data\\dummy'
    procs = -1   
    augment_object = img_augmentor(ori_img_folder,ori_csv,aug_type,out_folder,procs) 
    augment_object.augmentor_main()
    


    
