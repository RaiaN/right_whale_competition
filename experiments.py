#!/usr/bin/env python
__author__ = 'Alexandra Vesloguzova, Peter Leontiev, Sergey Krivohatskiy'
from skimage.io import imread
from skimage import color
from sklearn import ensemble
from collections import OrderedDict
import datetime
import pickle
import random
import os
import multiprocessing

SUBMISSION_FILENAME = 'submission.csv'


def prepare_clf(foreground_whale_id, background_whale_id, foreground_imgs, background_imgs):
    print("%s, %s" % (foreground_whale_id, background_whale_id))

    ind = 0
    f_len = len(foreground_imgs)
    while len(background_imgs) < f_len:
        background_imgs.append(background_imgs[ind])
        ind += 1

    ind = 0
    b_len = len(background_imgs)
    while len(foreground_imgs) < b_len:
        foreground_imgs.append(foreground_imgs[ind])
        ind += 1

    X = foreground_imgs + background_imgs
    Y = [foreground_whale_id] * len(foreground_imgs) + [background_whale_id] * len(background_imgs)

    rf_clf = ensemble.RandomForestClassifier(n_estimators=100, class_weight='auto')
    #ab_clf = ensemble.AdaBoostClassifier(n_estimators=100, learning_rate=0.8)
    #gb_clf = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.8)

    rf_clf.fit(X, Y)
    #ab_clf.fit(X, Y)
    #gb_clf.fit(X, Y)

    return foreground_whale_id, rf_clf

def get_train_whale_ids():
    return [filename for filename in os.listdir("groups/") if os.path.isdir("groups/"+filename)]

def prepare_clfs():
    whale_ids = get_train_whale_ids()

    print("Reading images...")
    all_imgs = {}
    for wid in whale_ids:
        X = []
        for image_name in os.listdir("groups/" + wid):
            image = imread(os.path.join("groups", wid, image_name))
            if image.shape[-1] == 3:
                image = color.rgb2gray(image)
            X.append(image.ravel())   
        all_imgs[wid] = X     
    print("Done!")

    pool = multiprocessing.Pool(processes=4)

    results = []
    for whale_id in whale_ids:
        foreground_imgs = all_imgs[whale_id]
        foreground_len = len(foreground_imgs)

        background_imgs = []
        background_len = 0
        while background_len < foreground_len:
            background_whale_id = random.choice(whale_ids)
            if background_whale_id == whale_id:
                continue
            background_imgs.append(random.choice(all_imgs[background_whale_id]))
            background_len += 1

        print("Async task for %s..." % whale_id)
        results.append(pool.apply_async(prepare_clf, args=(whale_id, "BACKGROUND", foreground_imgs, background_imgs)))

    print("Learning...")
    binary_clfs = {}
    while len(results) > 0:
        foreground_whale_id, clfs = results.pop().get()
        binary_clfs[foreground_whale_id] = clfs
    pickle.dump(binary_clfs, open("processed/clfs.bin", "wb"))

    print("Done!")


def get_binary_clfs_by_pickle():
    binary_clfs = pickle.load(open("processed/clfs.bin", "rb"))
    return OrderedDict(sorted(binary_clfs.items(), key=lambda t: t[0]))


def predict_worker(q, binary_clfs, image_filename, x):
    print("Start processing %s" % image_filename)
    probs = [clf.predict_proba(x)[0][0] for _, clf in binary_clfs.items()]
    q.put((image_filename, probs))


def write_sub_sub(results):
    with open("submission.csv", "a") as outp:
        for test_image, probs in results:
            outp.write(test_image + "," + ",".join(str(val) for val in probs) + "\n")


def store(q):
    print(datetime.datetime.now())
    results = []
    while True:
        res = q.get()
        if res is None:
            break
        results.append(res)
        if len(results) == 50:
            print(datetime.datetime.now())
            print("Writing sub sub...")            
            write_sub_sub(results)
            results = []
    
    print(datetime.datetime.now())
    write_sub_sub(results)

def main():
    try:
        os.makedirs("processed")
    except:
        pass

    whale_dirs = [whale_dirs for _, whale_dirs, _ in os.walk("groups/")][0]
    train_images = sum([os.listdir("groups/" + whale_dir) for whale_dir in whale_dirs], [])
    print("Train images: %s" % len(train_images))
    
    for _, _, filenames in os.walk("pre_processing/region_crop_gray_downscale"):
        all_filenames = set(filenames)
    test_images = all_filenames.difference(train_images)
    
    del all_filenames
    del train_images

    print("Test images: %s" % len(test_images))
    
    if not os.path.exists("processed/clfs.bin"):
        prepare_clfs()

    binary_clfs = get_binary_clfs_by_pickle()
    whale_classes = ("whale_%s" % whale_id for whale_id in binary_clfs.keys())
    with open("submission.csv", "w") as outp:
        outp.write("Image," + ",".join(whale_classes) + "\n")    

    print("Reading test images...")
    tasks = []    
   
    q = multiprocessing.Manager().Queue()    
    pool = multiprocessing.Pool(processes=3)  
    pool.apply_async(store, args=(q,)) 

    for image_filename in test_images:
        image = imread(os.path.join("pre_processing/region_crop_gray_downscale/", image_filename))
        if image.shape[-1] == 3:
            image = color.rgb2gray(image)
        tasks.append(pool.apply_async(predict_worker, args=(q, binary_clfs, image_filename, image.ravel())))   

    print("Predicting...")
    for task in tasks:
        task.get()
    q.put(None) # finish 
    
    print("Done!")

    return 0

if __name__ == "__main__":
    main()