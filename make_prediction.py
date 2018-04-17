import numpy as np
import tensorflow as tf
from case_tf import CIFAR10_quick

def check_correct(prob, path):
    neg_prob, pos_prob= prob
    is_pos = path.find('_p_') != -1 # find '_p_' in the file name
    
    if not is_pos and is_pos == (pos_prob > neg_prob):
        print(prob, path, 'True negative')
    
    return is_pos == (pos_prob > neg_prob)

# load the converted mean file
means = np.load('mean.npy')
mean_tensor = tf.transpose(tf.convert_to_tensor(means, dtype=tf.float32), [1, 2, 0])

def classify():
    '''Classify the given images using GoogleNet.'''

    model_data_path = './case_tf.npy'

    image_file_name_pattern = './subs/*.png'
    
    NUM_OF_IMAGES = 100
    
    # according to the .prototxt
    IMAGE_SIZE = 32
    IMAGE_CHANNELS = 3
    
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))

    # Construct the network
    net = CIFAR10_quick({'data': input_node})

    # Create an image producer (loads and processes images in parallel)
#     image_producer = dataset.ImageProducer(image_paths=image_paths)

    # custom: read images
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(image_file_name_pattern))
    
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    
    my_img = tf.image.decode_png(value)
    
    

    with tf.Session() as sess:

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())  
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)


        print('Load weights...')
        net.load(data_path=model_data_path, session=sess)


        image_list = []
        image_path_list = []

        print('Making predictions...')
        
        for _ in range(0, NUM_OF_IMAGES):
            single_image = sess.run(my_img)
            

            # Note (3 April) convert image channel sequence from RGB to BGR
            reversed_image = tf.reverse(single_image, [-1])
            reversed_image = tf.cast(reversed_image, tf.float32)
            
            final_image = tf.subtract(reversed_image, mean_tensor)
            
            image_list.append(final_image)
            image_path_list.append(sess.run(key))
        
        input_images = sess.run(tf.stack(image_list))
        probs = sess.run(net.get_output(), feed_dict={input_node: input_images})
        
        acc_list = []
        predictions = zip(probs, image_path_list)
        for prob, path in predictions:
            acc_list.append(check_correct(prob, path))
            
        print('accuracy: {}'.format(acc_list.count(True) / float(len(acc_list))))
        
        for prob, path in predictions[:20]:
            print('Image: {}, prob: {}'.format(path, prob))

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=2)

if __name__ == '__main__':
    classify()