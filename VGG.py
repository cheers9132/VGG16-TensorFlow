import tensorflow as tf
import utils

slim = tf.contrib.slim


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):

    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        # 两层3X3卷积，特征层为64
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # 池化层
        net = slim.max_pool2d(net, [2,2], scope='pool1')

        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')

        net = slim.conv2d(net, 1000, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')

        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')

        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net


if __name__ == '__main__':
    img1 = utils.load_image('./test_data/dog.jpg')

    inputs = tf.placeholder(tf.float32, [None, None, 3])
    resized_img = utils.resize_image(inputs, (224, 224))

    prediction = vgg_16(inputs)

    sess = tf.Session()
    cpkt_filename = './model/vgg_16.ckpt'

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver
    saver.restore(sess, cpkt_filename)

    pro = tf.nn.softmax(prediction)
    pre = sess.run(pro, feed_dict={inputs: img1})

    print('result:')
    utils.print_prob(pre[0], './synset.txt')

