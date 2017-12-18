import tensorflow as tf


class Preprocessor:
    def __init__(self, train, centered=True, rescaled=True, grayscale=True, shaped=True):
        self.mean_image = tf.reduce_mean(train, axis=0)
        self.centered = centered
        self.rescaled = rescaled
        self.grayscale = grayscale
        self.shaped = shaped
        self.luminosity = tf.constant([0.21, 0.72, 0.07],dtype=tf.float32)


    def apply(self, images, num_images):
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        
        if self.centered:
            images = images - self.mean_image

        if self.rescaled:
            images = images / 255

        if self.grayscale:
            gray = tf.tensordot(self.luminosity,images,axes=[[0],[1]])
            gray = tf.reshape(gray,shape=[num_images,1,32,32])
            images = tf.concat([images, gray], axis=1)

        if self.shaped:
            images = tf.transpose(images, perm=[0,2,3,1])

        return images

