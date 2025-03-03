from __future__ import print_function
from PIL import Image
import numpy as np
import time
import sys
from math import sqrt
import math
import json

http_loc="/data/data/com.termux/files/usr/share/apache2/default-site/htdocs/done.png"
out_loc="done.png"

unique_counter=0

def init_run(fname):
    """
        Identify points of temporary balance. 
        Big changes have big influence.
        Small changes have little influence.
        A balance is restored when there occurs a series of ups followed by 
        an equal no of downs or vice versa.
         color2:    ...  }
    """
    img=open_image(fname)
    img_cond=condition_img(img)
    sort_dict_and_write_json(img_cond,fname)   
    return img_cond.shape

def main():
    fname=sys.argv[1]
    #init_run(fname)
    blob_dict=read_json(fname)
    shape=read_shape(fname)
    done=new_image(shape)
    blob_dict_summary(blob_dict)
    blob=blob_dict[0][0]
    draw_pixels_i(blob,200,done)
    print("len(blob)=",len(blob))
    sym_dict={}
    
    write_image(out_loc,done)
    
    

 get_unique_counter():
    global unique_counter
    unique_counter+=1
    return unique_counter


def get_corner_pixels(blob,shape):
    c=list()
    for i in range(1,len(blob)-1):
        p1=i_to_p(blob[i-1],shape)
        p2=i_to_p(blob[i+1],shape)
        if p1[0]!=p2[0] and p1[1]!=p2[1]:
            c.append(blob[i])
    return c

def per_dist(a,b,c):
    return abs((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))/sqrt(pow(a[1]-b[1],2)+pow(a[0]-b[0],2))

def per_dist_i(i1,i2,i3,shape):
    p1=i_to_p(i1,shape)
    p2=i_to_p(i2,shape)
    p3=i_to_p(i3,shape)
    return per_dist(p1,p2,p3)

def angle(a,b):
    """slope of line ab"""
    ang=math.degrees(math.atan2(a[1]-b[1],a[0]-b[0]))
    #return int(round(ang))
    return ang

def angle_i(i1,i2,shape):
    """slope of line ab"""
    a=i_to_p(i1,shape)
    b=i_to_p(i2,shape)
    return angle(a,b)

def get_slope_i(i1,i2,shape):
    p1=i_to_p(i1,shape)
    p2=i_to_p(i2,shape)
    return get_slope(p1,p2)

def len_between(p1,p2):
    if(p1[0]==p2[0]):
        return abs(p1[1]-p2[1])
    else: return abs(p1[0]-p2[0])

def len_between_i(i1,i2,shape):
    p1=i_to_p(i1,shape)
    p2=i_to_p(i2,shape)
    return len_between(p1,p2)

def dist(p1,p2):
    return sqrt(float(pow(p2[0]-p1[0],2)+pow(p2[1]-p1[1],2)))

def dist_i(i1,i2,shape):
    p1=i_to_p(i1,shape)
    p2=i_to_p(i2,shape)
    return dist(p1,p2)

def sort_blob_mix(blob_mix,shape):
    blank_img=new_image(shape)
    temp_blob_mix=blob_mix[:]
    draw_pixels_i(temp_blob_mix,15,blank_img)
    blobs=list()
    blobs.append(list())
    i=0
    count=0
    blobs[i].append(temp_blob_mix[-1])
    while 1:
        count+=1
        if(get_color_i(get_north_i(blobs[i][-1],shape[1]),blank_img)==15):
            draw_pixel_i(blobs[i][-1],200,blank_img)
            temp_blob_mix.remove(blobs[i][-1])
            blobs[i].append(get_north_i(blobs[i][-1],shape[1]))
        elif(get_color_i(get_east_i(blobs[i][-1],shape[1]),blank_img)==15):
            draw_pixel_i(blobs[i][-1],200,blank_img)
            temp_blob_mix.remove(blobs[i][-1])
            blobs[i].append(get_east_i(blobs[i][-1],shape[1]))
        elif(get_color_i(get_south_i(blobs[i][-1],shape[1]),blank_img)==15):
            draw_pixel_i(blobs[i][-1],200,blank_img)
            temp_blob_mix.remove(blobs[i][-1])
            blobs[i].append(get_south_i(blobs[i][-1],shape[1]))
        elif(get_color_i(get_west_i(blobs[i][-1],shape[1]),blank_img)==15):
            draw_pixel_i(blobs[i][-1],200,blank_img)
            temp_blob_mix.remove(blobs[i][-1])
            blobs[i].append(get_west_i(blobs[i][-1],shape[1]))
        else:
            draw_pixel_i(blobs[i][-1],200,blank_img)
            temp_blob_mix.remove(blobs[i][-1])
            blobs[i].reverse()
            i+=1
            if(count==len(blob_mix)):
                break
            blobs.append(list())
            blobs[i].append(temp_blob_mix[-1])
    print(len(blobs),"blob found.")
    return blobs
     
def condition_img(img):
    img_scaled=np.array(Image.new('L',(img.shape[1]*3,img.shape[0]*3),(255)))
    for r in range(0,img.shape[0]):
        for c in range(0,img.shape[1]):
            img_scaled[r*3][c*3]=img[r][c]
            img_scaled[r*3][c*3+1]=img[r][c]
            img_scaled[r*3][c*3+2]=img[r][c]
            img_scaled[r*3+1][c*3]=img[r][c]
            img_scaled[r*3+1][c*3+1]=img[r][c]
            img_scaled[r*3+1][c*3+2]=img[r][c]
            img_scaled[r*3+2][c*3]=img[r][c]
            img_scaled[r*3+2][c*3+1]=img[r][c]
            img_scaled[r*3+2][c*3+2]=img[r][c]
    return img_scaled


def get_color(p,img):
    return img[p[0]][p[1]]
def get_color_i(pi,img):
    r=int(pi/img.shape[1])
    c=pi%img.shape[1]
    return img[r][c]
def get_north(p):
    return [p[0]-1,p[1]]
def get_east(p):
    return [p[0],p[1]+1]
def get_south(p):
    return [p[0]+1,p[1]]
def get_west(p):
    return [p[0],p[1]-1]
def pixel_equals(p1,p2):
    if p1[0]==p2[0] and p1[1]==p2[1]:
        return 1
    return 0
def get_north_i(pi,width):
    r=int(pi/width)
    c=pi%width
    return (r-1)*width+c
def get_east_i(pi,width):
    r=int(pi/width)
    c=pi%width
    return r*width+c+1
def get_south_i(pi,width):
    r=int(pi/width)
    c=pi%width
    return (r+1)*width+c
def get_west_i(pi,width):
    r=int(pi/width)
    c=pi%width
    return r*width+c-1
def i_to_p(i,shape):
    r=int(i/shape[1])
    c=i%shape[1]
    return [r,c]
def p_to_i(p,shape):
    return p[0]*shape[1]+p[1]

def angle_between(a,b,c):
    """angle between lines ab and bc"""
    ang=math.degrees(math.atan2(c[1]-b[1],c[0]-b[0])-math.atan2(a[1]-b[1],a[0]-b[0]))
    #return int(round(ang))
    return ang

def angle_between_i(ai,bi,ci,shape):
    a=i_to_p(ai,shape)
    b=i_to_p(bi,shape)
    c=i_to_p(ci,shape)
    return angle_between(a,b,c)

def next_i(i,pixels):
    if(i==len(pixels)-1):
        return 0
    else: return i+1

def prev_i(i,pixels):
    if i==0:
        return len(pixels)-1
    else: return i-1

def step(index,pixels,s):
    i=index
    if(s>0):
        ss=s
        while(ss>0):
            i=next_i(i,pixels)
            ss-=1
    else:
        ss=abs(s)
        while(ss>0):
            i=prev_i(i,pixels)
            ss-=1
    return i


def read_image(img,blob_dict):
    """ blob_dict={color: [[pixels],[pixels]]"""
    for r in range(img.shape[0]-1):
        for c in range(img.shape[1]-1):
            if img[r][c]!=img[r][c+1]: # left ro right
                if int(img[r][c]) in blob_dict:
                    blob_dict[int(img[r][c])].add(r*img.shape[1]+c)
                else:
                    blob_dict[int(img[r][c])]=set()
                    blob_dict[int(img[r][c])].add(r*img.shape[1]+c)


                if int(img[r][c+1]) in blob_dict:
                    blob_dict[int(img[r][c+1])].add(r*img.shape[1]+c+1)
                else:
                    blob_dict[int(img[r][c+1])]=set()
                    blob_dict[int(img[r][c+1])].add(r*img.shape[1]+c+1)

            if img[r][c]!=img[r+1][c]: # top to bottom
                if int(img[r][c]) in blob_dict:
                    blob_dict[int(img[r][c])].add(r*img.shape[1]+c)
                else:
                    blob_dict[int(img[r][c])]=set()
                    blob_dict[int(img[r][c])].add(r*img.shape[1]+c)


                if int(img[r+1][c]) in blob_dict:
                    blob_dict[int(img[r+1][c])].add((r+1)*img.shape[1]+c)
                else:
                    blob_dict[int(img[r+1][c])]=set()
                    blob_dict[int(img[r+1][c])].add((r+1)*img.shape[1]+c)

            if img[r][c]!=img[r+1][c+1]: # diagonal
                if int(img[r][c]) in blob_dict:
                    blob_dict[int(img[r][c])].add(r*img.shape[1]+c)
                else:
                    blob_dict[int(img[r][c])]=set()
                    blob_dict[int(img[r][c])].add(r*img.shape[1]+c)


                if int(img[r+1][c+1]) in blob_dict:
                    blob_dict[int(img[r+1][c+1])].add((r+1)*img.shape[1]+c+1)
                else:
                    blob_dict[int(img[r+1][c+1])]=set()
                    blob_dict[int(img[r+1][c+1])].add((r+1)*img.shape[1]+c+1)

            if img[r+1][c]!=img[r][c+1]: # diagonal
                if int(img[r+1][c]) in blob_dict:
                    blob_dict[int(img[r+1][c])].add((r+1)*img.shape[1]+c)
                else:
                    blob_dict[int(img[r+1][c])]=set()
                    blob_dict[int(img[r+1][c])].add((r+1)*img.shape[1]+c)

                if int(img[r][c+1]) in blob_dict:
                    blob_dict[int(img[r][c+1])].add(r*img.shape[1]+c+1)
                else:
                    blob_dict[int(img[r][c+1])]=set()
                    blob_dict[int(img[r][c+1])].add(r*img.shape[1]+c+1)
    for c,blobs in blob_dict.items():
        blob_dict[c]=list(blobs)
    print("unsorted dict created.")


def blob_dict_summary(blob_dict):
    n=0
    for key,val in blob_dict.items():
        print (key,len(val))
        n+=len(val)
    print(len(blob_dict),"colors found.")
    print(n,"bounds found.")

def write_image(fname,image):
    Image.fromarray(image).save(fname)

def open_image(fname):
    """returns the image as numpy array"""
    return np.array(Image.open(fname).convert('L'))
def new_image(shape):
    return np.array(Image.new('L',(shape[1],shape[0]),(255)))

def get_pixels_from(s,e,blob):
    pixels=list()
    while 1:
        pixels.append(blob[s])
        s=next_i(s,blob)
        if s==e:
            pixels.append(blob[s])
            break
    return pixels

def draw_pixel(p,i,img):
    if(get_color(p,img)==i):
        print("alert: re-draw")
    else: 
        img[p[0]][p[1]]=i
def draw_pixel_i(p,i,img):
    r=int(p/img.shape[1])
    c=p%img.shape[1]
    if(img[r][c]==i):
        print("alert: re-draw")
    else: img[r][c]=i

def draw_blobs_i(blobs,i1,i2,img):
    for even in range(0,len(blobs),2):
        draw_pixels_i(blobs[even],i1,img)
    for odd in range(1,len(blobs),2):
        draw_pixels_i(blobs[odd],i2,img)

def sort_dict_and_write_json(img,fname):
    start=time.time()
    blob_dict={}
    read_image(img,blob_dict)
    for c,val in blob_dict.items():
        blob_dict[c]=sort_blob_mix(blob_dict[c],img.shape)
    write_json(blob_dict,fname+".sort_dict.txt")
    write_json(img.shape,fname+".shape.txt")
    print ("sorted in",time.time()-start,"s.")

def write_json(blob_dict,fname):
    json.dump(blob_dict,open(fname,'w'))
    print(fname,"written.")

def read_shape(fname):
    return json.load(open(fname+".shape.txt"))

def read_json(fname):
    """ note that the key is read as string."""
    raw_read=json.load(open(fname+".sort_dict.txt"))
    sort_dict={}
    for k,v in raw_read.items():
        sort_dict[int(k)]=v
    return sort_dict

def draw_blob_dict(blob_dict,shape,out_location):
    new_img=new_image(shape)
    for k,blobs in blob_dict.items():
        for blob in blobs:
            draw_pixels(blob,k+1,new_img)
    write_image(out_location,new_img)

def draw_pixels_len(pixels,s,l,c,img):
    ss=s
    ee=step(ss,pixels,l)
    draw_pixel_i(pixels[ss],c,img)
    while 1:
        ss=next_i(ss,pixels)
        draw_pixel_i(pixels[ss],c,img)
        if ss==ee:
            break

def draw_pixels(pixels,i,img):
    for p in pixels:
        draw_pixel(p,i,img)
def draw_pixels_i(pixels,i,img):
    for p in pixels:
        draw_pixel_i(p,i,img)


if __name__=="__main__":
    main()
