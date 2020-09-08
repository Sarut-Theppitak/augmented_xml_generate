# augmented_xml_generate
Generate augmented images and xml files from original images and an original CSV file with specified a number of processors to use.

This python scrpit takes the path to original images to generate augmented images as well as takes a original csv file to make a new a set of xml files coresponding to the newly created images.

Currently, only 3 types of augmentation is available.

1.flipVer

2.flipHor

3.enhanced(ImageEnhance.Contrast(im).enhance(1.3))

    One should change the input parameters under the __name__ == '__main__': in the script:

    ori_img_folder >> path to original image folder

    ori_csv >> full path of the proginal csv file

    aug_type >> augmentation type (curently only 3 : flipVer, flipHor, enhanced )

    out_folder >> Specify your output folder. The augmented img folder,the coresponding xml folder,and the coresponding csv file will be created in this folder.

    procs >> number of processor to se (-1 means maximum processor available)


Ex:

    ori_img_folder = 'C:\\Users\\3978\\Desktop\\Faster-Rcnn\\inceptionV2_ais_s1_20200826\\data\\sample_img'

    ori_csv = 'C:\\Users\\3978\\Desktop\\Faster-Rcnn\\inceptionV2_ais_s1_20200826\\data\\sample.csv'

    aug_type = 'flipVer'

    out_folder = 'C:\\Users\\3978\\Desktop\\Faster-Rcnn\\inceptionV2_ais_s1_20200826\\data\\dummy'

    procs = -1   

