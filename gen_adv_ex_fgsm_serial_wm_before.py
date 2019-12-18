#!/usr/bin/env python3

from make_mnist_cnn_tf import build_cnn_mnist_model, reset_graph
import tensorflow as tf
import numpy as np
import time
import argparse
import cv2
import re
##start

# for i in range(0, test_length):
    # for j in range(0,6):
        # td_str+=str(test_d[i].item(j))
        # td_str+=" "        

def prep_for_targeted(x_train,target_value, lighten, origanal):
    if target_value == 0:
        # for i in range(len(x_train)):
            # x_train[i] = x_train[i]  - x_train[i]*lighten
            # x_train[i] = x_train[i] + x_train[1]*(lighten)
            
        x_train = x_train  - x_train*lighten
        x_train = x_train + origanal[1]*(lighten)

    if target_value == 1:        
        # for i in range(len(x_train)):
            # x_train[i] = x_train[i]  - x_train[i]*lighten
            # x_train[i] = x_train[i] + x_train[3]*(lighten)
            
        x_train = x_train  - x_train*lighten
        x_train = x_train + origanal[3]*(lighten)
    if target_value == 2:        
        # for i in range(len(x_train)):
            # x_train[i] = x_train[i]  - x_train[i]*lighten
            # x_train[i] = x_train[i] + x_train[5]*(lighten)
            
        x_train = x_train  - x_train*lighten
        x_train = x_train + origanal[5]*(lighten)
    if target_value == 3:        
        # for i in range(len(x_train)):
            # x_train[i] = x_train[i]  - x_train[i]*lighten
            # x_train[i] = x_train[i] + x_train[7]*(lighten)
            
        x_train = x_train  - x_train*lighten
        x_train = x_train + origanal[7]*(lighten)
    if target_value == 4:        
        # for i in range(len(x_train)):
            # x_train[i] = x_train[i]  - x_train[i]*lighten
            # x_train[i] = x_train[i] + x_train[9]*(lighten)
            
        x_train = x_train  - x_train*lighten
        x_train = x_train + origanal[9]*(lighten)
    if target_value == 5:        
        # for i in range(len(x_train)):
            # x_train[i] = x_train[i]  - x_train[i]*lighten
            # x_train[i] = x_train[i] + x_train[0]*(lighten)

        x_train = x_train  - x_train*lighten
        x_train = x_train + origanal[0]*(lighten)


    if target_value == 6:        
        # for i in range(len(x_train)):
            # x_train[i] = x_train[i]  - x_train[i]*lighten
            # x_train[i] = x_train[i] + x_train[13]*(lighten)

        x_train = x_train  - x_train*lighten
        x_train = x_train + x_train*(lighten)
    if target_value == 7:        
        # for i in range(len(x_train)):
            # x_train[i] = x_train[i]  - x_train[i]*lighten
            # x_train[i] = x_train[i] + x_train[15]*(lighten)

        x_train = x_train  - x_train*lighten
        x_train = x_train + origanal[15]*(lighten)


    if target_value == 8:        
        # for i in range(len(x_train)):
            # x_train[i] = x_train[i]  - x_train[i]*lighten
            # x_train[i] = x_train[i] + x_train[17]*(lighten)
            
        x_train = x_train  - x_train*lighten
        x_train = x_train + origanal[17]*(lighten)
    if target_value == 9:        
        # for i in range(len(x_train)):
            # x_train[i] = x_train[i]  - x_train[i]*lighten
            # x_train[i] = x_train[i] + x_train[4]*(lighten)            
    
        x_train = x_train  - x_train*lighten
        x_train = x_train + origanal[4]*(lighten)            

    return x_train
##end

def prep_for_targeted_before(x_train,target_value, lighten):
    if target_value == 0:
        for i in range(len(x_train)):
            x_train[i] = x_train[i]  - x_train[i]*lighten
            x_train[i] = x_train[i] + x_train[1]*(lighten)
            
        #x_train = x_train  - x_train*lighten
        #x_train = x_train + origanal[1]*(lighten)

    if target_value == 1:        
        for i in range(len(x_train)):
            x_train[i] = x_train[i]  - x_train[i]*lighten
            x_train[i] = x_train[i] + x_train[3]*(lighten)
            
        #x_train = x_train  - x_train*lighten
        #x_train = x_train + origanal[3]*(lighten)
    if target_value == 2:        
        for i in range(len(x_train)):
            x_train[i] = x_train[i]  - x_train[i]*lighten
            x_train[i] = x_train[i] + x_train[5]*(lighten)
            
        #x_train = x_train  - x_train*lighten
        #x_train = x_train + origanal[5]*(lighten)
    if target_value == 3:        
        for i in range(len(x_train)):
            x_train[i] = x_train[i]  - x_train[i]*lighten
            x_train[i] = x_train[i] + x_train[7]*(lighten)
            
        #x_train = x_train  - x_train*lighten
        #x_train = x_train + origanal[7]*(lighten)
    if target_value == 4:        
        for i in range(len(x_train)):
            x_train[i] = x_train[i]  - x_train[i]*lighten
            x_train[i] = x_train[i] + x_train[9]*(lighten)
            
        # x_train = x_train  - x_train*lighten
        # x_train = x_train + origanal[9]*(lighten)
    if target_value == 5:        
        for i in range(len(x_train)):
            x_train[i] = x_train[i]  - x_train[i]*lighten
            x_train[i] = x_train[i] + x_train[0]*(lighten)

        # x_train = x_train  - x_train*lighten
        # x_train = x_train + origanal[0]*(lighten)


    if target_value == 6:        
        for i in range(len(x_train)):
            x_train[i] = x_train[i]  - x_train[i]*lighten
            x_train[i] = x_train[i] + x_train[13]*(lighten)

        # x_train = x_train  - x_train*lighten
        # x_train = x_train + x_train*(lighten)
    if target_value == 7:        
        for i in range(len(x_train)):
            x_train[i] = x_train[i]  - x_train[i]*lighten
            x_train[i] = x_train[i] + x_train[15]*(lighten)

        # x_train = x_train  - x_train*lighten
        # x_train = x_train + origanal[15]*(lighten)


    if target_value == 8:        
        for i in range(len(x_train)):
            x_train[i] = x_train[i]  - x_train[i]*lighten
            x_train[i] = x_train[i] + x_train[17]*(lighten)
            
        # x_train = x_train  - x_train*lighten
        # x_train = x_train + origanal[17]*(lighten)
    if target_value == 9:        
        for i in range(len(x_train)):
            x_train[i] = x_train[i]  - x_train[i]*lighten
            x_train[i] = x_train[i] + x_train[4]*(lighten)            
    
        # x_train = x_train  - x_train*lighten
        # x_train = x_train + origanal[4]*(lighten)            

    return x_train
##end


def results_targeted(class_adv, class_y, zero_count,one_count,two_count,three_count,four_count,five_count,six_count,seven_count,eight_count, nine_count):
    if class_adv != class_y[0] and class_adv == 0:#trent
        zero_count = zero_count +1
    if class_adv != class_y[0] and class_adv == 1:#trent
        one_count = one_count +1
    if class_adv != class_y[0] and class_adv == 2:#trent
        two_count = two_count +1
    if class_adv != class_y[0] and class_adv == 3:#trent
        three_count = three_count +1
    if class_adv != class_y[0] and class_adv == 4:#trent
        four_count = four_count +1
    if class_adv != class_y[0] and class_adv == 5:#trent
        five_count = five_count +1
    if class_adv != class_y[0] and class_adv == 6:#trent
        six_count = six_count +1
    if class_adv != class_y[0] and class_adv == 7:#trent
        seven_count = seven_count +1
    if class_adv != class_y[0] and class_adv == 8:#trent
        eight_count = eight_count +1
    if class_adv != class_y[0] and class_adv == 9:#trent
        nine_count = nine_count +1        
    return zero_count,one_count,two_count,three_count,four_count,five_count,six_count,seven_count,eight_count, nine_count;

#file5 = open("test.txt", "w+")
file5 = open("test_before.txt", "w+")
#file5 = open("FGSM.txt", "w+")
for num in range(1):
#num = 3
    for i in range(10):
        L = 0.11 + 0.05*i

        parser = argparse.ArgumentParser()
        parser.add_argument('--epsmin', type=float, default=0.01)
        parser.add_argument('--epsmax', type=float, default=0.2)
        parser.add_argument('--idx', type=int, default=100)
        parser.add_argument('--numgens', type=int, default=1000)

        args = parser.parse_args()

        reset_graph()
        x = tf.placeholder(tf.float32, shape=(None, 28, 28))
        y = tf.placeholder(tf.int32, shape=(None,))
        model = build_cnn_mnist_model(x, y, False)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train / np.float32(255)
        y_train = y_train.astype(np.int32)
        x_test = x_test / np.float32(255)
        y_test = y_test.astype(np.int32)

        #print(x_train[1], y_train[1])
        #for i in range(100):
        #    print(y_train[i])

        grad, = tf.gradients(model['loss'], x)
        epsilon = tf.placeholder(tf.float32)
        optimal_perturbation = tf.multiply(tf.sign(grad), epsilon)
        adv_example_unclipped = tf.add(optimal_perturbation, x)

        adv_example = tf.clip_by_value(adv_example_unclipped, 0.0, 1.0)

        classes = tf.argmax(model['probability'], axis=1)

        adv_examples = []
        adv_targeted_count_0 = 0 #trent
        adv_targeted_count_1 = 0 #trent
        adv_targeted_count_2 = 0 #trent
        adv_targeted_count_3 = 0 #trent
        adv_targeted_count_4 = 0 #trent
        adv_targeted_count_5 = 0 #trent
        adv_targeted_count_6 = 0 #trent
        adv_targeted_count_7 = 0 #trent
        adv_targeted_count_8 = 0 #trent
        adv_targeted_count_9 = 0 #trent
        temp_0 = 0
        temp_1 = 0
        temp_2 = 0
        temp_3 = 0
        temp_4 = 0
        temp_5 = 0
        temp_6 = 0
        temp_7 = 0
        temp_8 = 0
        temp_9 = 0

        idx = args.idx
        epsilon_range = (args.epsmin, args.epsmax)

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        with tf.Session(config=config) as sess:
            saver.restore(sess, './models/mnist_cnn_tf/mnist_cnn_tf')
            acc_test = model['accuracy'].eval(feed_dict={
                x: x_test,
                y: y_test,
            })
            
            x_train = prep_for_targeted_before(x_train, num, L )
            # for i in range(len(x_train)):
                # x_train[i] = x_train[i]  - x_train[i]*0.9
                # x_train[i] = x_train[i] + x_train[1]*0.6
                
            print('Accuracy of model on test data: {}'.format(acc_test))
            print('Correct Class: {}'.format(y_train[idx]))
            class_x = classes.eval(feed_dict={x: x_train[idx:idx + 1]})
            print('Predicted class of input {}: {}'.format(idx, class_x))
            start = time.time()    
            for i in range(args.numgens):
                adv = adv_example.eval(
                    feed_dict={
                        x: x_train[idx:idx + 1],
                        y: y_train[idx:idx + 1],
                        epsilon: np.random.uniform(
                            epsilon_range[0], epsilon_range[1],
                            size=(28, 28)
                            )
                    })

                class_adv = classes.eval(feed_dict={x: adv})
                #class_adv = classes.eval(feed_dict={x: prep_for_targeted(adv, num, L, x_train)})

                if class_adv != y_train[0]:
                    adv_examples += [adv]
                    #print(class_adv)
                    temp_0, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8, temp_9 = results_targeted(class_adv, y_train, adv_targeted_count_0, adv_targeted_count_1, adv_targeted_count_2, adv_targeted_count_3, adv_targeted_count_4, adv_targeted_count_5, adv_targeted_count_6, adv_targeted_count_7, adv_targeted_count_8, adv_targeted_count_9)
                    adv_targeted_count_0 = temp_0
                    adv_targeted_count_1 = temp_1
                    adv_targeted_count_2 = temp_2
                    adv_targeted_count_3 = temp_3
                    adv_targeted_count_4 = temp_4
                    adv_targeted_count_5 = temp_5 
                    adv_targeted_count_6 = temp_6
                    adv_targeted_count_7 = temp_7 
                    adv_targeted_count_8 = temp_8
                    adv_targeted_count_9 = temp_9           
        print('Duration (s): {}'.format(time.time() - start))
        
        if adv_examples != []:
            adv_examples = np.concatenate(adv_examples, axis=0)
            print('Found {} adversarial examples.'.format(adv_examples.shape[0]))
            print('Percentage true adversarial examples: {}'.format(adv_examples.shape[0]/args.numgens))
        
            avg = np.zeros_like(x_train[idx])
            for i in range(adv_examples.shape[0]):
                avg += adv_examples[i]
            avg /= adv_examples.shape[0]
            stddev = 0
            for i in range(adv_examples.shape[0]):
                stddev += np.linalg.norm((adv_examples[i] - avg).flatten())

            stddev /= adv_examples.shape[0]
            print('Found std dev: {}'.format(stddev))
        print("number ", num,"L", L)
        file5.write("number " + str(num) + " L " + str(L) + "\n")   
        
        print('generated ', adv_targeted_count_0, ' zeros')#trent
        file5.write('generated ' +str(adv_targeted_count_0) + ' zeros\n')
        print('generated ', adv_targeted_count_1,' ones')#trent
        file5.write('generated ' + str(adv_targeted_count_1) +' ones\n')#trent
        print('generated ', adv_targeted_count_2,' twos')#trent
        file5.write('generated '+ str(adv_targeted_count_2) +' twos\n')#trent
        print('generated ', adv_targeted_count_3,' threes')#trent
        file5.write('generated '+ str(adv_targeted_count_3) +' threes\n')#trent
        print('generated ', adv_targeted_count_4,' fours')#trent
        file5.write('generated '+ str(adv_targeted_count_4)+' fours\n')#trent 
        print('generated ', adv_targeted_count_5,' fives')#trent
        file5.write('generated '+ str(adv_targeted_count_5)+' fives\n')#trent
        print('generated ', adv_targeted_count_6,' sixes')#trent
        file5.write('generated '+ str(adv_targeted_count_6)+' sixes\n')#trent
        print('generated ', adv_targeted_count_7,' sevens')#trent
        file5.write('generated '+ str(adv_targeted_count_7)+' sevens\n')#trent
        print('generated ', adv_targeted_count_8,' eights')#trent
        file5.write('generated '+ str(adv_targeted_count_8)+' eights\n')#trent
        print('generated ', adv_targeted_count_9,' nines')#trent
        file5.write('generated '+ str(adv_targeted_count_9)+' nines\n')#trent

file5.close() 

