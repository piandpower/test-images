import os
import urllib
import cv2
import matplotlib.pyplot as plt

dirname = os.path.dirname
url_dir = os.path.join(dirname(dirname(__file__)),'atf_images')
print(url_dir)

url_list_file = os.path.join(url_dir,'image_urls.txt')
cred_file = os.path.join(url_dir,'credentials.txt')
image_dir = os.path.join(url_dir,'images')
with open(cred_file,'r') as c:
    creds = c.readline().rstrip() # rstrip strips the newline character
    with open(url_list_file,'r') as f:
        for i, line in enumerate(f):
            line = line.rsplit('\n',1)[0] # strip newline character

            url = line.split('//',1)[0] + '//' + creds +line.split('//',1)[1]
            image = url.rsplit('/',1)[1]

            img_path = os.path.join(image_dir,image)
            urllib.urlretrieve(url,img_path)