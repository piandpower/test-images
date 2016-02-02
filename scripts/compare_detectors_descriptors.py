import os

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

image_pairs = [
    {
        "image1":  {"filename":"../original_images/01.jpg"},
        "image2":  {"filename":"../original_images/02.jpg"}
    }
]

both_list = ['brisk','akaze','kaze','orb']#,'sift','surf']
det_list = ['fast','agast','gftt','mser','star']
desc_list = ['freak','latch','lucid']

def findAllCombs(a_list,b_list,ab_list):
    combs = []
    for a in a_list:
        for b in b_list:
            combs.append((a,b))
        for ab in ab_list:
            combs.append((a,ab))
    for b in b_list:
        for ab in ab_list:
            combs.append((ab,b))
    for ab in ab_list:
        combs.append((ab,ab))
    return combs

# detectors are for finding keypoints.  They often also support computing
detectors = {"fast": cv2.FastFeatureDetector_create(), # lots of points (2159) all over
             "brisk": cv2.BRISK_create(),
             "akaze": cv2.AKAZE_create(),  # strange delocalization
             "kaze": cv2.KAZE_create(),  # strange delocalization
             "agast": cv2.AgastFeatureDetector_create(),
             "gftt": cv2.GFTTDetector_create(),
             "mser": cv2.MSER_create(),  # very few keypoints (80)
             "orb": cv2.ORB_create(),
             "star": cv2.xfeatures2d.StarDetector_create(),
             "sift": cv2.xfeatures2d.SIFT_create(),
             "surf": cv2.xfeatures2d.SURF_create(),
            }
descriptors_only = {
    "freak": cv2.xfeatures2d.FREAK_create(),
    "latch": cv2.xfeatures2d.LATCH_create(),
    "lucid": cv2.xfeatures2d.LUCID_create(1, 1),
}

def compute_features(image, detector_alg, descriptor_alg=None):
    data = image["image"]
    if descriptor_alg in detectors:
        kps, descriptors = detectors[descriptor_alg].detectAndCompute(data, None)
    elif descriptor_alg in descriptors_only:
        kps = detectors[detector_alg].detect(data)
        kps, descriptors = descriptors_only[descriptor_alg].compute(data, kps)
    else:
        raise ValueError("unknown algorithm passed to descriptor stage")
    image["kps"] = kps
    image["descriptors"] = descriptors
    return image

def perspective_match(reference, unknown, use_flann=False, min_match_count=10,descriptor=None):
    if use_flann:
        FLANN_INDEX_KDTREE = 0
        FLANN_INDEX_LSH    = 6
        # floating point algorithms
        if descriptor in ["sift", "surf"]:
            index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                                trees = 5)
        # binary algorithms
        else:
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 1) #2
        search_params = dict(checks = 50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(reference["descriptors"],
                               unknown["descriptors"],
                               k=2)
    good = []
    matchesMask=None
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>min_match_count:
        src_pts = np.float32([ reference["kps"][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ unknown["kps"][m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # this limits matches to being within the identified subimage
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
        except AttributeError:
            matchesMask, good = None, None

    else:
        print "Not enough matches are found (%d/%d)" % (len(good), min_match_count)
        matchesMask, good = None, None
    return matchesMask, good

def draw_matches(reference_features, unknown_features, mask, good_pts):
    fig = plt.figure()
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = (255,0,0),
                       matchesMask = mask,
                       flags = 2)

    img3 = cv2.drawMatches(reference_features["image"],
                           reference_features["kps"],
                           unknown_features["image"],
                           unknown_features["kps"],
                           good_pts,None,**draw_params)
    plt.imshow(img3)
    return fig

def wrapper(reference, unknown, detector_alg,
            descriptor_alg=None, use_flann=False,
            min_match_count=5):
    if not descriptor_alg:
        descriptor_alg = detector_alg
    reference_features = compute_features(reference,
                                          detector_alg,
                                          descriptor_alg)
    unknown_features = compute_features(unknown,
                                        detector_alg,
                                        descriptor_alg)
    print('ref features: ',len(reference_features['kps']))
    print('unknown features: ',len(unknown_features['kps']))
    matchesMask, good_pts = perspective_match(reference_features,
                                              unknown_features,
                                             use_flann=use_flann,
                                             min_match_count=min_match_count,
                                             descriptor=descriptor_alg)
    fig = draw_matches(reference_features, unknown_features,
                 matchesMask, good_pts)
    fig.gca().set_title("keypoints: {}, detector: {}, Matcher: {}".format(
        detector_alg, descriptor_alg,
        "FLANN" if use_flann else "Brute Force"))
    if good_pts is not None:
        return len(reference_features['kps']),len(unknown_features['kps']),len(good_pts)
    else:
        return len(reference_features['kps']),len(unknown_features['kps']),0



def compareDetectorsDescriptors(image_file1,image_file2,outfile=None):
    images = {
        "image1": {"filename":image_file1},
        "image2": {"filename":image_file2}
    }

    for image in images:
        images[image]["image"] = cv2.imread(images[image]["filename"])
        #images[image]["image"] = cv2.cvtColor(images[image]["image"], cv2.COLOR_BGR2RGB)

    df = pd.DataFrame(findAllCombs(det_list,desc_list,both_list),
                  columns=['detector','descriptor'])
    df['combo'] = df['detector']+df['descriptor']
    df = df.set_index('combo')
    for row in df.index:
        print(row)
        print(df.loc[row,'detector'])
        print(df.loc[row,'descriptor'])
        df.loc[row,'img1_kps'],df.loc[row,'img2_kps'],df.loc[row,'num_matches'] = wrapper(images["image1"],
                                            images["image2"],
                                            detector_alg=df.loc[row,'detector'],
                                            descriptor_alg=df.loc[row,'descriptor'])#,
                                            #use_flann=True)
        pct_match = df.loc[row,'num_matches']/min(df.loc[row,'img1_kps'],df.loc[row,'img2_kps'])
        df.loc[row,'pct_match'] = pct_match
        print(df.loc[row,'num_matches'])
        print('-'*40)
    df.sort_values('num_matches', ascending=False).to_csv(outfile)

for image_pair in image_pairs:
    image1_name = os.path.split(image_pair['image1']['filename'])[-1].split('.')[0]
    image2_name = os.path.split(image_pair['image2']['filename'])[-1].split('.')[0]
    outfile="../results/results_"+image1_name+'_vs_'+image2_name+'.csv'
    compareDetectorsDescriptors('../original_images/01.jpg',
                            '../original_images/02.jpg',
                            outfile=outfile)