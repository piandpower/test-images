from scripts.compare_detectors_descriptors import compareDetectorsDescriptors

image_pairs = [
    {
        "image1":  {"filename":"../kris_images/Pistol_22/IMG_6082.jpg"},
        "image2":  {"filename":"../kris_images/Pistol_22/IMG_6079.jpg"}
    },
    {
        "image1":  {"filename":"../original_images/01.jpg"},
        "image2":  {"filename":"../original_images/05.png"}
    },
    {
        "image1":  {"filename":"../kris_images/Winchester_small/IMG_6121.JPG_resizeXpct.jpg"},
        "image2":  {"filename":"../kris_images/Winchester_small/IMG_6122.JPG_resizeXpct.jpg"}
    }
]

for image_pair in image_pairs:
    image1_name = os.path.split(image_pair['image1']['filename'])[-1].split('.')[0]
    image2_name = os.path.split(image_pair['image2']['filename'])[-1].split('.')[0]
    outfile="../results/results_"+image1_name+'_vs_'+image2_name+'.csv'
    compareDetectorsDescriptors('../original_images/01.jpg',
                            '../original_images/02.jpg',
                            outfile=outfile)