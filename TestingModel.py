import TSPORTNetcv2
import numpy as np
import TSPORTNet
import os
import sys
import tensorflow as tf
import time
import vgg16


def load_img_list(dataset):

    if dataset == 'ECSSD':
        path = '/home/hpc/LY/dataset/imgs/ECSSD'
    elif dataset == 'HKU-IS-TE':
        path = '/home/hpc/LY/dataset/imgs/HKU-IS-TE'
    elif dataset == 'SOD':
        path = '/home/hpc/LY/dataset/imgs/SOD'
    elif dataset == 'PASCAL-S':
        path = '/home/hpc/LY/dataset/imgs/pascal'
    elif dataset == 'DUT-OMRON':
        path = '/home/hpc/LY/dataset/imgs/DUT-OMRON'
    elif dataset == 'DUTS-TE':
        path = '/home/hpc/LY/dataset/imgs/DUTS-TE'

    imgs = os.listdir(path)

    return path, imgs


if __name__ == "__main__":

    model = TSPORTNet.Model()
    model.build_model()
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    img_size = TSPORTNet.img_size
    label_size = TSPORTNet.label_size

    ckpt = tf.train.get_checkpoint_state('Model/')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)

    #datasets = ['MSRA-B', 'HKU-IS', 'DUT-OMRON',
     #           'PASCAL-S', 'ECSSD', 'SOD']
    datasets = ['ECSSD']   

    if not os.path.exists('Result'):
        os.mkdir('Result')

    for dataset in datasets:
        path, imgs = load_img_list(dataset)

        save_dir = 'Result/' + dataset
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_dir = 'Result/' + dataset
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for f_img in imgs:

            img = cv2.imread(os.path.join(path, f_img))
            img_name, ext = os.path.splitext(f_img)

            if img is not None:
                ori_img = img.copy()
                img_shape = img.shape
                img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
                img = img.reshape((1, img_size, img_size, 3))

                start_time = time.time()
                #result = sess.run(model.Prob,
                #                  feed_dict={model.input_holder: img,
                #                             model.keep_prob: 1})
                result = sess.run(model.Prob,
                                              feed_dict={model.input_holder: img,
                                                         })                
                print("--- %s seconds ---" % (time.time() - start_time))

                result = np.reshape(result, (label_size, label_size, 2))
                result = result[:, :, 0]

                result = cv2.resize(np.squeeze(result), (img_shape[1], img_shape[0]))

                save_name = os.path.join(save_dir, img_name+'.png')
                cv2.imwrite(save_name, (result*255).astype(np.uint8))

    sess.close()
