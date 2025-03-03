from __future__ import print_function
from PIL import Image
import numpy as np
import time
import sys
from math import sqrt
import math
import json
from operator import sub

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
    

    # point, hline, vline, sline, curve, shape, object and scene.
    # p1,p2,p3,p4,p5,p6,p7,p8,p9,p10 ... p307200
    # hl1,hl2,hl3,hl4,hl5
    # vl1,vl2,vl3,vl4,vl5
    # sl1,sl2,sl3,sl4
    # c1,c2,c3
    # s1,s2
    # obj1
    
    """ 
    1. RAW DATA FROM SENSOR Eq. MICROPHONE INPUT
    2. CHANGE IN SENSORY DATA. (FIRST ORDER)
    3. CHANGE IN CHANGE IN SENSORY DATA. (SECOND ORDER)
    4. CHANGE IN CHANGE IN CHANGE IN SENSORY DATA. (THIRD ORDER)
    5. CHANGE IN CHANGE IN ... (NTH ORDER)

    REVERSE MAPPING 
    SYMBOL : MEANING OF THE WORD
    MEANING : SYMBOL
    OF : SYMBOL
    THE : SYMBOL
    WORD : SYMBOL

    LINE : LINE1, LINE2, LINE3 ...
    LENGTH : L1, L2, L3, L4, ...
    L1 : LINE1,LINE2,LINE4,...
    L2 : LINE3,LINE5,LINE6,...
    L3 : LINE10,LINE11,LINE12,...
    SLOPE : S1, S2, S3, S4, ...
    S1 : LINE2,LINE3,LINE4,...
    S2 : LINE1,LINE11,LINE12,...
    S3 : LINE5,LINE21,LINE22,...
    POSITION : I1,I2,I3,I4, ...
    I1 : LINE1,LINE2,LINE3,...
    I2 : LINE4,LINE5,LINE6,...
    I3 : LINE10,LINE11,LINE12,...
    POINT : P1,P2,P3,P4,...
    COLOR : C1,C2,C3,
    C1 : P1,P2,P3,P4,...
    C2 : P10,P11,P12,...
    C3 : P20,P21,P22,...
    X-COORDINATE : X1,X2,X3,...C
    Y-COORDINATE : Y1,Y2,Y3,...R
    X1 : P1,P2,P3,...
    X2 : P10,P11,P21,...
    Y1 : P20,P21,P22,...
    Y2 : P30,P31,P32,...

    IT SHOULD BE POSSIBILE TO EXPRESS AND RECOGNIZE A PATTERN WITH EXCESS
    AMOUNT OF DATA SUCH THAT THE BLUEPRINT REMAINS THE SAME.

    THE EXCESS DATA IS REDUCED TO A BASIC UNDERSTANDING OF THE PATTERN.


    LINE(SLOPE,LENGTH,POSITION)
    Lines of similar slope, length, or positon are grouped.

    REPRESENT GROUP OF PIXELS WITH A UNIQUE IDENTIFIER/SYMBOL.
    POINTS OF CHANGE IN DIRECTION AND HOW LONG IN THAT DIRECTION.
    
    C
    B
    A
    T
    CA - CAT
    AT - CAT BAT
    BA - BAT

    vline and hline are primary symbols.
    size of hline and vline relative to the size of the canvas
    influences whether it is perceived as a line or a point.
    
    displacement is considered in visual perception not distance.
    
    imagine that every word is a selector of subsets from a set.
    for eq. word 'top' divides the set into to two and selects
    the upper data.

    nothing is perfect. so is a sloping line. vline and hline can
    be perfect theoretically but a sloping line is always a 
    combination of vline and hline. every symbol is expressed as 
    a confidence level of its true definition.
    
    symbol LINE has properties vis LENGTH, SLOPE and POSITION.
    LINE : LINE-1 - LINE-N
    LINE-X : LENGTH-X,SLOPE-X,POSITION-X
    LENGTH : LENGTH-1 - LENGTH-
    LENGTH-X : LINE-X
    SLOPE-X : LINE-X
    POSITION-X : LINE-X
    
    every symbol has a blueprint
    blueprints are made of existing symbols
    so when the symbols VERTICAL LINE and HORIZONTAL LINE are close to perpendicular
    the data speaks what it is. its features speaks out loud to the world what it is.
    then occurs coincidence of a set of features which is recognized as objects.
    
    features are pushed into memories.
    
    EVIDENCE IN NO SPECIFIC ORDER.
    CLOSED FIGURE IS AN OBJECT WHERE THE ORDER IS IRRELEVANT.

    
    V4,H3,V5,H4 etc

    POWER OF OBSERVING EVIDENCES IN THE INPUT
    THAT GIVE RISE THE QUESTION, WHAT IS AN EVIDENCE?
    IN SIMPLE TERMS AN EVIDENCE IS A PATTERN
    AND PATTERN IS INPUT THAT FOLLOWS DEFINITE SIGNATURE.
    MEANS, SOMETHING THAT REPEATS WITH DEFINITE CHANGES.
    THE DEEPEST QUESTION IS - WHAT IS AN EVIDENCE?
    WHY DO WE RECOGNIZE EVIDENCE? 
    EVERY PIXEL IN AN IMAGE IS TAGGED TO AN EVIDENCE.
    COLOR IS AN EVIDENCE. POSITION IS AN EVIDENCE.
    GROUP OF PIXELS ARE TAGGED TO A SINGLE EVIDENCE.
    RICHNESS OF EXPERIENCE IS A CONSEQUENCE OF THE RICHNESS
    OF EVIDENCE TAGGED.


    RESOLUTION OF PERCEPTION REMAINS THE SAME. 
    THAT IS, IF THE INPUT IS 500 POINTS AND THE RESOLUTION 
    OF BRAIN ANALYZING THE INPUT IS 10 THEN THE DETAILS ARE
    50 POINTS APART. THUS, IF THE INPUT IS 100 POINTS THEN
    THE DETAILS ARE 100/10 THAT IS 10 POINTS.
    THE RESOLVING POWER OF THE BRAIN REMAINS THE SAME REGARDLESS OF 
    THE INPUT COMPLEXITY. BRAIN IS LIKE A MACRO LENS.

    Effectively, sensory data stream is nothing but a series of valleys and
    curves, big and small.
    """
    fname=sys.argv[1]
    fname="rect-small.png"
    start_time=time.time()
    #init_run(fname)
    blob_dict=read_json(fname)
    shape=read_shape(fname)
    print("shape: ",shape)
    done=new_image(shape)
    blob=blob_dict[255][0][:1000:5]
    draw_pixels_i(blob,200,done)
    print("len(blob)=",len(blob))
    sym_dict={}


    i=int(sys.argv[1])
    #result=get_pair_full_nonc(blob,shape)
    #write_json(result,fname+".pair.txt")
    res=read_json_all(fname+".pair.txt")
    p=i
    s=res[p][0]
    e=step(s,blob,res[p][1])
    draw_pixels_i(get_pixels_from(s,e,blob),40,done)
    write_image(out_loc,done)
    print("finished in",time.time()-start_time,"seconds.")



def is_parent_blob(ip_blob,blob):
    for i in blob:
        if i not in ip_blob:
            return 0
    return 1
def is_child_blob(ip_blob,blob):
    for i in ip_blob:
        if i not in blob:
            return 0
    return 1

def is_child(sl_ip,sl,blob):
    """
    checks if [s_ip,l_ip] is child of [s,l]
    """
    ip_blob=get_pixels_from(sl_ip[0],
            step(sl_ip[0],blob,sl_ip[1]),blob)
    blob=get_pixels_from(sl[0],
            step(sl[0],blob,sl[1]),blob)   
    return is_child_blob(ip_blob,blob)

def is_parent(sl_ip,sl,blob):
    ip_blob=get_pixels_from(sl_ip[0],
            step(sl_ip[0],blob,sl_ip[1]),blob)
    blob=get_pixels_from(sl[0],
            step(sl[0],blob,sl[1]),blob)   
    return is_parent_blob(ip_blob,blob)

def has_pair(s,e,blob,shape):
    np=list()
    i=s
    last=0
    if not on_same_side(s,e,blob,shape):
        return False
    while(1):
        i=next_i(i,blob)
        if i==e:
            np.append(-1)
            break

        pd=abs(per_dist_i(blob[s],blob[e],blob[i],shape))
        if pd>last:
            np.append(1)
        elif pd<last:
            np.append(-1)
        else:
            return False
            np.append(0)
        last=pd
    np2=list()
    np2.append(np[0])
    for i in range(1,len(np)):
        if np2[-1]!=np[i]:
            np2.append(np[i])
    if np2.count(1)>1 or np2.count(-1)>1 or np2.count(0)>1:
        return False
    return True



def get_pair_full_nonc(pixel_group,shape):
    res=list()
    for i in range(3,len(pixel_group),1):
        res1=get_pair_blob_nonc(i,pixel_group,shape)
        for r in res1:
            res.append(r)
    
    temp=list()
    found_child=False
    for r in res:
        found_child=False
        for rr in res:
            if is_child(r,rr,pixel_group) and r!=rr:
                found_child=True
                break
        if not found_child:
            temp.append(r)
            

    temp=sorted(temp,key=lambda x: x[0],reverse=True)            
    return temp
 

def get_pair_blob_nonc(l,pixel_group,shape):
    indices=list()
    for s in range(0,len(pixel_group)-l):
        e=s+l-1
        if has_pair(s,e,pixel_group,shape):
            indices.append([s,l])
    return indices


def get_npair_full(blob,shape):
    res=list()
    for i in range(3,100,1):
        res1=get_npair_blob(i,blob,shape)
        for r in res1:
            res.append(r)
    res=sorted(res,key=lambda x: x[1],reverse=True)
    return res

def get_npair_blob(l,blob,shape):
    s=0
    e=step(s,blob,l-1)
    indices=list()
    step_val=1
    n=step_val
    while 1:
        np=has_npair(s,e,blob,shape)
        if np:
            indices.append([n-step_val,l])
        s=step(s,blob,step_val)
        e=step(e,blob,step_val)
        if n>=len(blob):
            break
        n+=step_val
    return indices


def find_sym_objects_nonc(blob,shape):
    found=list()
    ip_blobs=[blob]
    while len(ip_blobs)>0:
        ip_blob=ip_blobs.pop(-1)
        #print("len(ip_blob)=",len(ip_blob))

        if len(ip_blob)<4:
            print("found len=",len(ip_blob))
            found.append(ip_blob)
            continue
        res=get_symmetry_index_full_nonc(ip_blob,shape)
        s=res[0]
        e=s+res[1]-1
        if s>0 and e<(len(ip_blob)-1):
            print("found normal at",res)
            found.append(ip_blob[s:e+1])
            ip_blobs.append(ip_blob[:s])
            ip_blobs.append(ip_blob[e+1:])
        elif s==0 and e==(len(ip_blob)-1):
            print("found full at",res)
            found.append(ip_blob)
        elif s==0:
            print("found left at",res)
            found.append(ip_blob[:e+1])
            ip_blobs.append(ip_blob[e+1:])
        elif e==(len(ip_blob)-1):
            print("found right at",res)
            found.append(ip_blob[:s])
            ip_blobs.append(ip_blob[:s-1])

    # make objects continuous
    s=0
    l=0
    tot_len=0
    for f in found:
        tot_len+=len(f)
    print("tot_len=",tot_len)
    found_sorted=list()
    while s<tot_len:
        for f in found:
            if f[0]==blob[s]:
                found_sorted.append(f)
                break
        s+=len(found_sorted[-1])

    return found_sorted


def get_symmetry_index_full_nonc(pixel_group,shape):
    res=list()
    for i in range(3,len(pixel_group),1):
        res1=get_symmetry_index_blob_nonc(i,pixel_group,shape)
        for r in res1:
            res.append(r)


    return res

def get_best_index(index_list):
    #get sym objects with same index value order
    #by decreasing length
    index_list=sorted(index_list,key=lambda x: x[0])
    best_index=index_list[0][0]
    order_by_length=list()
    for r in index_list:
        if r[0]==best_index:
            order_by_length.append(r)
    order_by_length=sorted(order_by_length,key=lambda x: x[2],reverse=True)
    return order_by_length
    return [order_by_length[0][1],order_by_length[0][2]]


def get_symmetry_index_blob_nonc(l,pixel_group,shape):
    indices=list()
    if l>=len(pixel_group):
        index=get_symmetry_index(0,len(pixel_group)-1,pixel_group,shape)
        #index=(index/len(pixel_group))*100
        
        tot_dist=total_dist(0,len(pixel_group)-1,pixel_group,shape)+1
        index=(index/tot_dist)*100
        return [index,0,len(pixel_group)]
    for s in range(0,len(pixel_group)-l):
        e=s+l-1
        index=get_symmetry_index(s,e,pixel_group,shape)
        #index=(index/l)*100
        tot_dist=total_dist(s,e,pixel_group,shape)+1
        index=(index/tot_dist)*100
        indices.append([index,s,l])
    return indices


def get_symmetry_index_full(blob,shape):
    res=list()
    for i in range(3,len(blob),1):
        res1=get_symmetry_index_blob(i,blob,shape)
        for r in res1:
            res.append(r)
    res=sorted(res,key=lambda x: x[0])
    return res
    #order by decreasing length among best index
    best_index=res[0][0]
    order_by_length=list()
    for r in res:
        if r[0]==best_index:
            order_by_length.append(r)
    order_by_length=sorted(order_by_length,key=lambda x: x[2],reverse=True)
    return [order_by_length[0][1],order_by_length[0][2]]

def get_symmetry_index_blob(l,blob,shape):
    s=0
    e=step(s,blob,l-1)
    indices=list()
    step_val=1
    n=step_val
    while 1:
        index=get_symmetry_index(s,e,blob,shape)
        #index=(index/l)*100 #optional
        indices.append([index,n-step_val,l])
        s=step(s,blob,step_val)
        e=step(e,blob,step_val)
        if n>=len(blob):
            break
        n+=step_val
    return indices


def get_symmetry_index(s,e,blob,shape):
    res=-1
    l=next_i(s,blob)
    if l==e:
        return 0
    r=prev_i(e,blob)
    if l==r:
        l_dist=dist_i(blob[s],blob[l],shape)
        r_dist=dist_i(blob[r],blob[e],shape)
        diff=abs(l_dist-r_dist)
        return diff
    while 1:
        l_dist1=dist_i(blob[s],blob[l],shape)
        l_dist2=dist_i(blob[s],blob[r],shape)
        r_dist1=dist_i(blob[l],blob[e],shape)
        r_dist2=dist_i(blob[r],blob[e],shape)
        diff1=abs(l_dist1-r_dist2)
        diff2=abs(l_dist2-r_dist1)
        if diff1>res:
            res=diff1
        if diff2>res:
            res=diff2
        l=next_i(l,blob)
        if l==r:
            return res
        if next_i(l,blob)==r:
            diff=abs(per_dist_i(blob[s],blob[e],blob[l],shape))
            if diff>res:
                res=diff
            l_dist=dist_i(blob[s],blob[l],shape)
            r_dist=dist_i(blob[l],blob[e],shape)
            diff=abs(l_dist-r_dist)
            if diff>res:
                return diff
            else:
                return res
        r=prev_i(r,blob)

def total_dist(s,e,blob,shape):
    dist=0
    pre=s
    while 1:
        cur=next_i(pre,blob)
        if cur==e:
            dist+=dist_i(blob[pre],blob[e],shape)
            break
        dist+=dist_i(blob[pre],blob[cur],shape)
        pre=cur
    return dist

def is_sym(start,end,pixels,threshold,shape):
    l=next_i(start,pixels)
    r=prev_i(end,pixels)
    if l==r:
        if abs((dist_i(pixels[start],pixels[l],shape))-\
                (dist_i(pixels[l],pixels[end],shape)))<threshold or per_dist_i(pixels[start],pixels[end],pixels[l],
                        shape)<threshold:
            return per_dist_i(pixels[start],pixels[end],pixels[l],shape)
        else: return -1
    while 1:
        if abs((dist_i(pixels[start],pixels[l],shape))-\
                (dist_i(pixels[r],pixels[end],shape)))>threshold and per_dist_i(pixels[start],pixels[end],pixels[l],
                        shape)>threshold:
                    return -1
        if abs((dist_i(pixels[l],pixels[end],shape))-\
                (dist_i(pixels[start],pixels[r],shape)))>threshold and per_dist_i(pixels[start],pixels[end],pixels[r],
                    shape)>threshold:
                    return -1
        l=next_i(l,pixels)
        if l==r:
            return per_dist_i(pixels[start],pixels[end],pixels[l],shape)
        if next_i(l,pixels)==r:
            if abs((dist_i(pixels[start],pixels[l],shape))-\
                (dist_i(pixels[l],pixels[end],shape)))<threshold or per_dist_i(pixels[start],pixels[end],pixels[l],
                        shape)<threshold:
                return per_dist_i(pixels[start],pixels[end],pixels[l],shape)
            else: return -1
        r=prev_i(r,pixels)
 

def carry_forward(lst):
    cf_array=list()
    cf=lst[-1]+lst[0]
    cf_array.append(cf)
    for i in range(1,len(lst)):
        cf=cf+lst[i]
        cf_array.append(cf)
    return cf_array

def count_zeros(lst):
    c=0
    for i in range(0,len(lst)):
        if lst[i]==0:
            c+=1
    return c

def count_nonzeros(lst):
    c=0
    for i in range(0,len(lst)):
        if lst[i]!=0:
            c+=1
    return c


def get_add_array(lst):
    add_array=list()
    a=lst[-1]+lst[0]
    add_array.append(a)
    for i in range(1,len(lst)):
        add_array.append(lst[i-1]+lst[i])
    return add_array


def remove_multi_zero(lst):
    no_multi_zero=list()
    no_multi_zero.append(lst[0])
    for i in range(1,len(lst)):
        if lst[i]==0 and no_multi_zero[-1]==0:
            pass
        else:
            no_multi_zero.append(lst[i])
    return no_multi_zero


def compare_array_deep(lst1,lst2):
    if len(lst1)!=len(lst2):
        return -1
    rotated=lst2
    max_diff=compare_array(lst1,rotated)
    min_diff=max_diff
    min_diff_array=lst2
    for i in range(1,len(lst1)):
        rotated=rotate_array_cw(rotated)
        max_diff=compare_array(lst1,rotated)
        if max_diff<min_diff:
            min_diff=max_diff
            min_diff_array=rotated
    print(min_diff)
    return min_diff_array

def compare_array(lst1,lst2):
    if len(lst1)!=len(lst2):
        return -1
    result=list(map(sub,lst2,lst1))
    return max(result)

def rotate_array_cw(lst):
    rotated=list()
    rotated.append(lst[-1])
    for i in range(0,len(lst)-1):
        rotated.append(lst[i])
    return rotated

def get_angle_diff(lst):
    diff_array=list()
    diff=angle_diff(lst[-1],lst[0])
    diff_array.append(diff)
    for i in range(1,len(lst)):
        diff=angle_diff(lst[i-1],lst[i])
        diff_array.append(diff)
    
    return diff_array

def get_angle_array(blob,shape):
    angle_array=list()
    a=angle_i(blob[-1],blob[0],shape)
    angle_array.append(a)
    for i in range(1,len(blob)):
        a=angle_i(blob[i-1],blob[i],shape)
        angle_array.append(a)
    return angle_array
 
def get_dist_array(blob,shape):
    dist_array=list()
    d=dist_i(blob[-1],blob[0],shape)
    dist_array.append(d)
    for i in range(1,len(blob)):
        d=dist_i(blob[i-1],blob[i],shape)
        dist_array.append(d)
    return dist_array


def get_unique_counter():
    global unique_counter
    unique_counter+=1
    return unique_counter

def is_increasing(s,e,blob,shape):
    # ab is line joining s and e
    # xy is line perpendicular to ab at point s
    # function checks if perp distance between xy and
    # all points between s and e is increasing or not.

    # find slope of line joining s,e
    p1=i_to_p(blob[s],shape)
    p2=i_to_p(blob[e],shape)
    slope=(float)(p2[0]-p1[0])/(float)(p2[1]-p1[1]+0.00000000000001)
    m=-1/(slope+0.00000000001)
    c=p1[0]-m*p1[1]

    pd_prev=0
    pd_list=list()
    i=next_i(s,blob)
    thresh=2
    pp=i_to_p(blob[i],shape)
    while 1:
        pd=(m*pp[1]-pp[0]+c)/sqrt(pow(m,2)+1)
        if (pd-pd_prev)>thresh:
            pd_list.append(1)
        elif (pd_prev-pd)>thresh:
            pd_list.append(-1)
        else:
            pd_list.append(0)
        pd_prev=pd
        i=next_i(i,blob)
        if i==e:
            break
        pp=i_to_p(blob[i],shape)
    #print(m,c)
    #print(p1,p2)
    print(pd_list)
    if pd_list.count(1)>0 and pd_list.count(-1)>0 or pd_list.count(0)>0:
        return 0
    else:
        return 1

def find_increasing_acw(s,blob,shape):
    n=5
    s_=step(s,blob,-5)
    while 1:
        if is_increasing(s_,s,blob,shape) and n<len(blob):
            n+=1
            s_=prev_i(s_,blob)
        else:
            break
    #print(i_between(next_i(s,blob),e,blob))
    return (next_i(s_,blob),s)

def find_increasing_cw(s,blob,shape):
    n=5
    e_=step(s,blob,5)
    while 1:
        if is_increasing(s,e_,blob,shape) and n<len(blob):
            n+=1
            e_=next_i(e_,blob)
        else:
            break
    #print(i_between(s,prev_i(e,blob),blob))
    return (s,prev_i(e_,blob))



def get_largest_line_quick(blob,shape):
    start=time.time()
    l1=len(blob)
    l3=3
    l2=int((l1+l3)/2)
    while 1:
        print(l1,l2,l3)
        if has_line_of_len(blob,l2,shape):
            if l1>l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        else:
            if l1<l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        if l1==l2 or l3==l2:
            break
    for s in range(0,len(blob)):
        e=step(s,blob,l2-1)
        if is_line(s,e,blob,shape):
            print("found quick line in",time.time()-start,"seconds. l=",l2)
            return (s,l2)

def is_line(s,e,blob,shape):
    max_pd=get_max_per_dist(s,e,blob,shape)
    if max_pd<(0.05*dist_i(blob[s],blob[e],shape)):
        return True
    return False

def has_line_of_len(blob,l,shape):
    for s in range(0,len(blob)):
        e=step(s,blob,l-1)
        if is_line(s,e,blob,shape):
            return True
    return False

def has_line_of_len_nonc(blob,l,shape):
    for s in range(0,len(blob)-l):
        e=step(s,blob,l-1)
        if is_line(s,e,blob,shape):
            return True
    return False

def get_largest_line_quick_nonc(blob,shape):
    start=time.time()
    l1=len(blob)
    l3=3
    l2=int((l1+l3)/2)
    while 1:
        print(l1,l2,l3)
        if has_line_of_len_nonc(blob,l2,shape):
            if l1>l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        else:
            if l1<l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        if l1==l2 or l3==l2:
            break
    for s in range(0,len(blob)-l2):
        e=step(s,blob,l2-1)
        if is_line(s,e,blob,shape):
            print("found quick line in",time.time()-start,"seconds. l=",l2)
            return (s,l2)



def get_max_per_dist(s,e,blob,shape):
    i=s 
    max_pd=0
    while 1:
        i=next_i(i,blob)
        if i==e:
            break
        pd=per_dist_i(blob[s],blob[e],blob[i],shape)
        if pd>max_pd:
            max_pd=pd
    return max_pd

    pairs=list()
    while 1:
        s,e=find_same_side_both(e,blob,shape)
        n+=1
        if (s,e) in pairs:
            break
        else:
            pairs.append((s,e))
    return pairs


def find_same_side_both(start,end,blob,shape):
    i_prev=0
    s=start
    e=end
    while 1:
        s,e=find_same_side_cw(s,e,blob,shape)
        s,e=find_same_side_acw(s,e,blob,shape)

        i=i_between(s,e,blob)
        if i==i_prev:
            break
        else:
            i_prev=i
    
    j_prev=0
    s_=start
    e_=end
    while 1:
        s_,e_=find_same_side_acw(s_,e_,blob,shape)
        s_,e_=find_same_side_cw(s_,e_,blob,shape)
        j=i_between(s_,e_,blob)
        if j==j_prev:
            break
        else:
            j_prev=j
    
    if i_prev>=j_prev:
        return (s,e)
    else:
        return (s_,e_)

def find_same_side_acw(s,e,blob,shape):
    n=0
    s_=s
    while 1:
        if on_same_side(s_,e,blob,shape) and n<len(blob):
            n+=1
            s_=prev_i(s_,blob)
        else:
            break
    #print(i_between(next_i(s,blob),e,blob))
    return (next_i(s_,blob),e)

def find_same_side_cw(s,e,blob,shape):
    n=0
    e_=e
    while 1:
        if on_same_side(s,e_,blob,shape) and n<len(blob):
            n+=1
            e_=next_i(e_,blob)
        else:
            break
    #print(i_between(s,prev_i(e,blob),blob))
    return (s,prev_i(e_,blob))

def get_outer_corner(blob,shape):
    outer=list()
    for i in range(1,len(blob)-1):
        a=i_to_p(blob[i-1],shape)
        b=i_to_p(blob[i+1],shape)
        c=i_to_p(blob[i],shape)
        vi=((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))
        if vi>0:
            outer.append(blob[i])
    a=i_to_p(blob[-2],shape)
    b=i_to_p(blob[0],shape)
    c=i_to_p(blob[-1],shape)
    vi=((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))
    if vi>0:
        outer.append(blob[-1])

    a=i_to_p(blob[-1],shape)
    b=i_to_p(blob[1],shape)
    c=i_to_p(blob[0],shape)
    vi=((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))
    if vi>0:
        outer.append(blob[0])

    return outer



def get_inner_corner(blob,shape):
    inner=list()
    for i in range(1,len(blob)-1):
        a=i_to_p(blob[i-1],shape)
        b=i_to_p(blob[i+1],shape)
        c=i_to_p(blob[i],shape)
        vi=((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))
        if vi<0:
            inner.append(blob[i])
    a=i_to_p(blob[-2],shape)
    b=i_to_p(blob[0],shape)
    c=i_to_p(blob[-1],shape)
    vi=((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))
    if vi<0:
        inner.append(blob[-1])

    a=i_to_p(blob[-1],shape)
    b=i_to_p(blob[1],shape)
    c=i_to_p(blob[0],shape)
    vi=((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))
    if vi<0:
        inner.append(blob[0])

    return inner


def on_same_side(s,e,blob,shape):
    a=i_to_p(blob[s],shape)
    b=i_to_p(blob[e],shape)
    i=s
    val=list()
    while 1:
        i=next_i(i,blob)
        if i==e:
            break
        c=i_to_p(blob[i],shape)
        vi=((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))
        if vi>0:
            val.append(1)
        elif vi<0:
            val.append(-1)
        else:
            val.append(0)

    #return val
    if val.count(1)>0 and val.count(-1)>0:
        return 0
    else:
        return 1

def get_npair_old(s,e,blob,shape):
    np=list()
    i=s
    last=0
    while(1):
        i=next_i(i,blob)
        if i==e:
            np.append(-1)
            break
        pd=int(per_dist_i(blob[s],blob[e],blob[i],shape))
        if pd>last:
            np.append(1)
        elif pd<last:
            np.append(-1)
        else:
            np.append(0)
        last=pd

    for i in range(0,len(np)-1):
        if np[i]!=0 and np[i+1]==0:
            np[i+1]=np[i]
    for i in range(len(np)-2,-1,-1):
        if np[i+1]!=0 and np[i]==0:
            np[i]=np[i+1]
    first=list()
    first.append(blob[s])
    for i in range(1,len(np)):
        if np[i]==-1 and np[i-1]==1:
            first.append(blob[step(s,blob,i)])
    first.append(blob[e])
    #print("found",len(first)-2,"npair[s].")
    return len(first)-2

def get_corner_pixels(blob,shape):
    c=list()
    for i in range(1,len(blob)-1):
        p1=i_to_p(blob[i-1],shape)
        p2=i_to_p(blob[i+1],shape)
        if p1[0]!=p2[0] and p1[1]!=p2[1]:
            c.append(blob[i])
    return c

def break_blob_curve_c(blob,threshold,shape):
    s,l=get_largest_curve_quick(blob,threshold,shape)
    return strip_between_c(s,step(s,blob,l),blob)

def has_sym_of_len_curve(blob,l,threshold,shape):            
    for s in range(0,len(blob)):
        e=step(s,blob,l-1)
        sl=is_sym(s,e,blob,threshold,shape)
        if sl>threshold:
            return True
    return False


def get_largest_curve_quick(blob,threshold,shape):
    start=time.time()
    l1=len(blob)
    l3=3
    l2=int((l1+l3)/2)
    while 1:
        print(l1,l2,l3)
        if has_sym_of_len(blob,l2,threshold,shape):
            if l1>l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        else:
            if l1<l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        if l1==l2 or l3==l2:
            break
    for s in range(0,len(blob)):
        e=step(s,blob,l2-1)
        sl=is_sym(s,e,blob,threshold,shape)
        if sl>threshold:
            print("found quick sym in",time.time()-start,"seconds. l=",l2)
            return (s,l2)


def get_largest_sym_quick(blob,threshold_,shape):
    start=time.time()
    l1=len(blob)
    l3=3
    l2=int((l1+l3)/2)
    threshold=threshold_
    while 1:
        #threshold=int(0.01*l2)
        print(l1,l2,l3,threshold)
        if has_sym_of_len(blob,l2,threshold,shape):
            if l1>l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        else:
            if l1<l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        if l1==l2 or l3==l2:
            break
    for s in range(0,len(blob)):
        e=step(s,blob,l2-1)
        #threshold=int(0.05*l2)
        sl=is_sym(s,e,blob,threshold,shape)
        if sl>=0:
            print("found quick sym in",time.time()-start,"seconds. l=",l2)
            return (s,l2)

def get_largest_sym_quick_nonc(blob,threshold_,shape):
    start=time.time()
    threshold=5
    l1=len(blob)
    l3=3
    l2=int((l1+l3)/2)
    while 1:
        print(l1,l2,l3)
        if has_sym_of_len_nonc(blob,l2,threshold,shape):
            if l1>l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        else:
            if l1<l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        if l1==l2 or l3==l2:
            break
    for s in range(0,len(blob)-l2):
        e=step(s,blob,l2-1)
        sl=is_sym(s,e,blob,threshold,shape)
        if sl>=0:
            print("found quick sym in",time.time()-start,"seconds. l=",l2)
            return (s,l2)


def has_sym_of_len(blob,l,threshold,shape):            
    for s in range(0,len(blob)):
        e=step(s,blob,l-1)
        sl=is_sym(s,e,blob,threshold,shape)
        if sl>=0:
            return True
    return False

def has_sym_of_len_nonc(blob,l,threshold,shape):
    for s in range(0,len(blob)-l):
        e=step(s,blob,l-1)
        sl=is_sym(s,e,blob,threshold,shape)
        if sl>=0 and sl<threshold:
            return True
    return False

def break_blob_c(blob,threshold,shape):
    s,l=get_largest_sym_quick(blob,threshold,shape)
    return strip_between_c(s,step(s,blob,l),blob)

def break_blob_nonc(blob,threshold,shape):
    s,l=get_largest_sym_quick_nonc(blob,threshold,shape)
    return strip_between_nonc(s,step(s,blob,l),blob)


def break_blobs_c(blobs,threshold,shape):
    super_blobs=list()
    for blob in blobs:
        blobs_=break_blob(blob,threshold,shape)
        for blob_ in blobs_:
            super_blobs.append(blob_)
        blobs_=list()
    return super_blobs

def break_blobs_nonc(blobs,threshold,shape):
    super_blobs=list()
    for blob in blobs:
        blobs_=break_blob_nonc(blob,threshold,shape)
        for blob_ in blobs_:
            super_blobs.append(blob_)
        blobs_=list()
    return super_blobs

def nth_break_blob_nonc(n,blob,threshold,shape):
    c=0
    super_blobs=break_blob_nonc(blob,threshold,shape)
    while 1:
        if c==n:
            return super_blobs
        else:
            super_blobs=break_blobs_nonc(super_blobs,threshold,shape)
            c+=1

def strip_between_c(s,e,blob):
    ss=s
    first=list()
    while 1:
        first.append(blob[ss])
        ss=next_i(ss,blob)
        if ss==e:
            first.append(blob[ss])
            break
    second=list()
    ee=next_i(e,blob)
    while 1:
        second.append(blob[ee])
        ee=next_i(ee,blob)
        if ee==s:
            break
    #return second
    return [first,second]

def strip_between_nonc(s,e,blob):
    blobs=list()
    
    temp=list()
    for i in range(0,s):
        temp.append(blob[i])
    blobs.append(temp)
 
    temp=list()
    for i in range(s,e+1):
        temp.append(blob[i])
    blobs.append(temp)
    
    temp=list()
    for i in range(e,len(blob)):
        temp.append(blob[i])
    blobs.append(temp)
 
    return blobs


def is_sym(start,end,pixels,threshold,shape):
    l=next_i(start,pixels)
    r=prev_i(end,pixels)
    if l==r:
        if abs((dist_i(pixels[start],pixels[l],shape))-\
                (dist_i(pixels[l],pixels[end],shape)))<threshold or per_dist_i(pixels[start],pixels[end],pixels[l],
                        shape)<threshold:
            return per_dist_i(pixels[start],pixels[end],pixels[l],shape)
        else: return -1
    while 1:
        if abs((dist_i(pixels[start],pixels[l],shape))-\
                (dist_i(pixels[r],pixels[end],shape)))>threshold and per_dist_i(pixels[start],pixels[end],pixels[l],
                        shape)>threshold:
                    return -1
        if abs((dist_i(pixels[l],pixels[end],shape))-\
                (dist_i(pixels[start],pixels[r],shape)))>threshold and per_dist_i(pixels[start],pixels[end],pixels[r],
                    shape)>threshold:
                    return -1
        l=next_i(l,pixels)
        if l==r:
            return per_dist_i(pixels[start],pixels[end],pixels[l],shape)
        if next_i(l,pixels)==r:
            if abs((dist_i(pixels[start],pixels[l],shape))-\
                (dist_i(pixels[l],pixels[end],shape)))<threshold or per_dist_i(pixels[start],pixels[end],pixels[l],
                        shape)<threshold:
                return per_dist_i(pixels[start],pixels[end],pixels[l],shape)
            else: return -1
        r=prev_i(r,pixels)
 

def i_between(start,end,pixels):
    c=1
    s=start
    while 1:
        s=next_i(s,pixels)
        if(s==end):
            break
        c+=1
    return c

def per_dist(a,b,c):
    return abs((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))/sqrt(pow(a[1]-b[1],2)+pow(a[0]-b[0],2))

def per_dist_i(i1,i2,i3,shape):
    p1=i_to_p(i1,shape)
    p2=i_to_p(i2,shape)
    p3=i_to_p(i3,shape)
    return per_dist(p1,p2,p3)

def find_angle_i(a,b,shape):
    p1=i_to_p(a,shape)
    p2=i_to_p(b,shape)
    return find_angle(p1,p2)

def find_angle(p1,p2):
    angle=math.atan2(p1[1]-p2[1],p1[0]-p2[0])*180/math.pi
    if angle<0:
        angle+=360
    return angle

def find_angle_diff(angle_pre,angle):
    angle_pre_q=0
    angle_q=0
    if angle<90:
        angle_q=1
    elif angle>=90 and angle<180:
        angle_q=2
    elif angle>=180 and angle<270:
        angle_q=3
    elif angle>=270 and angle<360:
        angle_q=4
    
    if angle_pre<90:
        angle_pre_q=1
    elif angle_pre>=90 and angle_pre<180:
        angle_pre_q=2
    elif angle_pre>=180 and angle_pre<270:
        angle_pre_q=3
    elif angle_pre>=270 and angle_pre<360:
        angle_pre_q=4
 
    if angle_pre_q==1:
        if angle_q==3 and (angle_pre+180)<angle:
            diff=360-angle+angle_pre
        elif angle_q==4:
            diff=360-angle+angle_pre
        else:
            diff=(angle-angle_pre)
    elif angle_pre_q==2:
        if angle_q==4 and (angle_pre+180)<angle:
            diff=360-angle+angle_pre
        else:
            diff=(angle-angle_pre)
    elif angle_pre_q==3:
        if angle_q==1 and (angle+180)<angle_pre:
            diff=360-angle_pre+angle
        else:
            diff=(angle-angle_pre)
    elif angle_pre_q==4:
        if angle_q==2 and (angle+180)<angle_pre:
            diff=360-angle_pre+angle
        elif angle_q==1:
            diff=360-angle_pre+angle
        else:
            diff=(angle-angle_pre)
    return diff

def angle_diff(a,b):
    diff=b-a
    if diff>180:
        diff=diff-360
    elif diff<-180:
        diff=diff+360
    return diff


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

def get_slope(p1,p2):
    slope=(float)(p2[0]-p1[0])/(float)(p2[1]-p1[1]+0.00001)
    return slope

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

def draw_pixels_force(pixels,i,img):
    for p in pixels:
        draw_pixel_force(p,i,img)
def draw_pixels_i_force(pixels,i,img):
    for p in pixels:
        draw_pixel_i_force(p,i,img)

def draw_pixel_force(p,i,img):
    if get_color(p,img)==i:
        print("force draw pixel")
        img[p[0]][p[1]]+=20
    else:
        img[p[0]][p[1]]=i
def draw_pixel_i_force(p,i,img):
    r=int(p/img.shape[1])
    c=p%img.shape[1]
    draw_pixel_force((r,c),i,img)

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

def read_json_all(fname):
    """ note that the key is read as string."""
    raw_read=json.load(open(fname))
    return raw_read


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
