import cv2
import numpy as np
import TSPORTNet
#import NLDFnew
import vgg16
import tensorflow as tf
import os
from config import cfg


def load_training_list():

    with open('/home/hpc/LY/NLDF/dataset/train/trainDUTS.txt') as f:
        lines = f.read().splitlines()

    files = []
    labels = []

    for line in lines:
        labels.append('%s' % line)
        files.append('%s' % line.replace('.png', '.jpg'))

    return files, labels


def load_train_val_list():

    files = []
    labels = []

    with open('/home/hpc/LY/NLDF/dataset/valid.txt') as f:
        lines = f.read().splitlines()

    for line in lines:
        labels.append('/home/hpc/LY/NLDF/dataset/valid_mask/%s' % line)
        files.append('/home/hpc/LY/NLDF/dataset/valid_img/%s' % line.replace('.png', '.jpg'))

    #with open('F:/related code/Non-Local Deep Features for Salient Object Detection (CVPR 2017)/NLDF-master/dataset/valid_imgs.txt') as f:
     #   lines = f.read().splitlines()

    #for line in lines:
     #   labels.append('dataset/MSRA-B/annotation/%s' % line)
      #  files.append('dataset/MSRA-B/image/%s' % line.replace('.png', '.jpg'))

    return files, labels


if __name__ == "__main__":

    model = TSPORTNet.Model()
    model.build_model()
    
    #modelnew = NLDFnew.Model()
    #modelnew.build_model()    
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    sess = tf.Session()

    global_step = 0
    max_grad_norm = 20
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(model.Loss_Mean, tvars), max_grad_norm)
    lr = 1e-5
    #lr = tf.train.exponential_decay(0.01, global_step, 1, 0.5, staircase = False)
    opt = tf.train.AdamOptimizer(lr)
    train_op = opt.apply_gradients(zip(grads, tvars))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    train_list, label_list = load_training_list()

    n_epochs = 5
    img_size = TSPORTNet.img_size
    label_size = TSPORTNet.label_size
    
    f = open("./Model/loss.txt", "w")
    f.truncate()
    f.close()    
    

    for i in range(n_epochs):     
        
        whole_loss = 0.0
        whole_acc = 0.0
        count = 0
        for f_img, f_label in zip(train_list, label_list):

            img = cv2.imread(f_img).astype(np.float32)
            img_flip0 = cv2.flip(img, 0)
            img_flip1 = cv2.flip(img, 1)
            img_flip2 = cv2.flip(img, -1)

            label = cv2.imread(f_label)[:, :, 0].astype(np.float32)
            label_flip0 = cv2.flip(label, 0)
            label_flip1 = cv2.flip(label, 1)
            label_flip2 = cv2.flip(label, -1)

            img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
            label = cv2.resize(label, (label_size, label_size))
            label = label.astype(np.float32) / 255.

            img = img.reshape((1, img_size, img_size, 3))
            label = np.stack((label, 1-label), axis=2)
            label = np.reshape(label, [-1, 2])
            
            if count==0:
                _, loss, acc, S11, S12, S21, S22 = sess.run([train_op, model.Loss_Mean, model.accuracy, model.S11, model.S12, model.S21, model.S22],
                                        feed_dict={model.input_holder: img,
                                                   model.label_holder: label,
                                                   model.S11_holder: np.ones([3*3*int(np.floor(cfg.B/2)), int(np.floor(cfg.C/2))], dtype=np.float32),
                                                   model.S12_holder: np.ones([3*3*int(np.floor(cfg.B/2)), int(np.floor(cfg.C/2))], dtype=np.float32),
                                                   model.S21_holder: np.ones([3*3*int(np.floor(cfg.C/2)), int(np.floor(cfg.D/2))], dtype=np.float32),
                                                   model.S22_holder: np.ones([3*3*int(np.floor(cfg.C/2)), int(np.floor(cfg.D/2))], dtype=np.float32)
                                                   })
            else:
                _, loss, acc, S11, S12, S21, S22 = sess.run([train_op, model.Loss_Mean, model.accuracy, model.S11, model.S12, model.S21, model.S22],
                                                                    feed_dict={model.input_holder: img,
                                                           model.label_holder: label,
                                                           model.S11_holder: S11,
                                                           model.S12_holder: S12,
                                                           model.S21_holder: S21,
                                                           model.S22_holder: S22
                                                           })                

            whole_loss += loss
            whole_acc += acc
            count = count + 1

            # add vertical flip image for training
            img_flip1 = cv2.resize(img_flip1, (img_size, img_size)) - vgg16.VGG_MEAN
            label_flip1 = cv2.resize(label_flip1, (label_size, label_size))
            label_flip1 = label_flip1.astype(np.float32) / 255.
            

            img_flip1 = img_flip1.reshape((1, img_size, img_size, 3))
            label_flip1 = np.stack((label_flip1, 1 - label_flip1), axis=2)
            label_flip1 = np.reshape(label_flip1, [-1, 2])

            _, loss, acc, S11, S12, S21, S22 = sess.run([train_op, model.Loss_Mean, model.accuracy, model.S11, model.S12, model.S21, model.S22],
                                    feed_dict={model.input_holder: img_flip1,
                                               model.label_holder: label_flip1,
                                               model.S11_holder: S11,
                                               model.S12_holder: S12,
                                               model.S21_holder: S21,
                                               model.S22_holder: S22
                                               })

            whole_loss += loss
            whole_acc += acc
            count = count + 1
            
            ## add horizon flip image for training
            #img_flip1 = cv2.resize(img_flip1, (img_size, img_size)) - vgg16.VGG_MEAN
            #label_flip1 = cv2.resize(label_flip1, (label_size, label_size))
            #label_flip1 = label_flip1.astype(np.float32) / 255.
        
        
            #img_flip1 = img_flip1.reshape((1, img_size, img_size, 3))
            #label_flip1 = np.stack((label_flip1, 1 - label_flip1), axis=2)
            #label_flip1 = np.reshape(label_flip1, [-1, 2])
        
            #_, loss, acc = sess.run([train_op, model.Loss_Mean, model.accuracy],
                                            #feed_dict={model.input_holder: img_flip1,
                                                       #model.label_holder: label_flip1})
        
            #whole_loss += loss
            #whole_acc += acc
            #count = count + 1
            
            ## add horizon and vertical flip image for training
            #img_flip2 = cv2.resize(img_flip2, (img_size, img_size)) - vgg16.VGG_MEAN
            #label_flip2 = cv2.resize(label_flip2, (label_size, label_size))
            #label_flip2 = label_flip2.astype(np.float32) / 255.
        
        
            #img_flip2 = img_flip2.reshape((1, img_size, img_size, 3))
            #label_flip2 = np.stack((label_flip2, 1 - label_flip2), axis=2)
            #label_flip2 = np.reshape(label_flip2, [-1, 2])
        
            #_, loss, acc = sess.run([train_op, model.Loss_Mean, model.accuracy],
                                            #feed_dict={model.input_holder: img_flip2,
                                                               #model.label_holder: label_flip2})
        
            #whole_loss += loss
            #whole_acc += acc
            #count = count + 1            

            if count % 1 == 0:
                print ("Loss of %d images: %f, Accuracy: %f" % (count, (whole_loss/count), (whole_acc/count)))

        #if whole_loss != whole_loss:
        #    lr = lr*0.1
        #    opt = tf.train.AdamOptimizer(lr)
        #    train_op = opt.apply_gradients(zip(grads, tvars))
        #    ckpt = tf.train.get_checkpoint_state("./Model")
        #    if ckpt and ckpt.model_checkpoint_path:
        #        print("Continue training from the model {}".format(ckpt.model_checkpoint_path))
        #        saver.restore(sess, ckpt.model_checkpoint_path)  
        #else:
        print ("Epoch %d: Loss: %f, Accuracy: %f " % (i+1, (whole_loss/count), (whole_acc/count)))
        saver.save(sess, 'Model/model.ckpt', global_step=i+1)
            
        f = open("./Model/loss.txt", "a+")
        #f.write("Lr: %f, Epoch %d: Loss: %f, Accuracy: %f \n" % (lr, i+1, (whole_loss/count), (whole_acc/count)))
        f.write("Epoch %d: Loss: %f, Accuracy: %f \n" % (i+1, (whole_loss/count), (whole_acc/count)))
        f.close()        
        
        

    #os.mkdir('Model')
    #saver.save(sess, 'Model/model.ckpt', global_step=n_epochs)
