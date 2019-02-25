from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
img_width, img_height = 150,150

model = load_model('C:/Users/vimal/Desktop/Coin classifier/models/coin.h5')
#model.summary()
def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(150, 150))
    plt.imshow(img)
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        
        plt.show()
        
    return img_tensor
 #image path
img_path ='C:/Users/vimal/Desktop/Coin classifier/testing/five rupee f 2.jpg'    # dir
#img_path = 'C:/Users/vimal/Desktop/one.jpg'      # desktop

    # load a single image
new_image = load_image(img_path)

    # check prediction
pred = model.predict_classes(new_image)
if(pred==0):
    print("One Rupee Coin")
elif(pred==1):
    print("Ten Rupee Coin")
elif(pred==2):
    print("Two Rupee Coin")
else:
    print("Five Rupee Coin")