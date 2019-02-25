
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

img_width, img_height = 150,150
datagen = ImageDataGenerator(rescale=1./255)
validation_data_dir = 'C:/Users/vimal/Desktop/Coin classifier/validate'
validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')
model = load_model('C:/Users/vimal/Desktop/dogscats/models/catdog2.h5')
#model.summary()
num_of_test_samples=139
batch_size = 16
Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)
y_pred = Y_pred > 0.5

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
target_names = ['One', 'Ten','Two','Five']
print('Classification report')
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

