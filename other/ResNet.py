import tensorflow as tf
class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet,self).__init__()
        self.block_1 = ResNetBlock()
        self.block_2 = ResNetBlock()
        self.global_pool = Layers.GlobalAveragePooling2D()
        
    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.global_pool(x)
        return self.classifier(x)
    
    resnet = ResNet()
    dataset = ...
    resnet.fit(dataset, epochs=10)
    

sentences = tf.ragged.constant([
    ["Hello", "World", "!"],
    ["We", "are", "testing","tf.ragged.constant", "."]
])

print(sentences)

