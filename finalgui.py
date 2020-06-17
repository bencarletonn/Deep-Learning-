from kivy.app import App
from kivy.graphics import Color, Ellipse 
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.stencilview import StencilView
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.lang import Builder
from functools import partial
import cv2
import matplotlib.pyplot as plt 
import os 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np




# Load the model 
class Net(nn.Module):

	def __init__(self):
		super().__init__()
		# 4 Linear hidden layers 
		self.fc1 = nn.Linear(784, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 64)
		self.fc4 = nn.Linear(64, 10)

	def forward(self, x):
		# Activation functions 
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.log_softmax(self.fc4(x), dim=1) 

		return x

model = Net()
model.load_state_dict(torch.load('./MNIST_net.pth'))
model.eval() 


'''
# Backgrounds of widgets 
Builder.load_string("""

<StencilTestWidget>:
	canvas.before:
        Rectangle:
            pos: self.pos
            size: self.size
            source: 'white.png'

""") 
'''


class StencilTestWidget(StencilView):


	def on_touch_down(self, touch):
		with self.canvas:
			Color(1, 1, 1)
			d = 30 
			Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))

	def on_touch_move(self, touch):
		with self.canvas:
			Color(1, 1, 1)
			d = 30 
			Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))


class StencilCanvasApp(App):
 

    def clear(self, label, wid, *largs):
        label.text = 'Prediction:'
        wid.canvas.clear()


    def take(self, wid, label, *largs):
    	Window.screenshot("number.png") 
    	PATH = os.path.join(os.getcwd() + "\\" + "number0001.png") 
    	img_array = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    	img_array = img_array[100:480, 160:622] # Crop the image
    	img_array = cv2.resize(img_array,(28,28))
    	os.remove(PATH) # Delete the saved image 
    	#plt.imshow(img_array) 
    	#plt.show() 
    	transform= transforms.Compose([transforms.ToTensor()])
    	img_array = transform(img_array)
    	output = model(img_array.view(-1,784))
    	output = torch.argmax(output)
    	#print(torch.argmax(output))
    	label.text = str(output.item())

      




    def build(self):
        wid = StencilTestWidget(size_hint=(None, None), size=[462.0, -390.0], pos=[160.0, 505.0])

        label = Label(text='Prediction:')

        btn_add500 = Button(text='Submit')
        btn_add500.bind(on_press=partial(self.take, wid, label))
        # Add button that links to CNN 

        btn_reset = Button(text='Clear')
        btn_reset.bind(on_press=partial(self.clear, label, wid))

        layout = BoxLayout(size_hint=(1, None), height=50)
        layout.add_widget(btn_add500)
        layout.add_widget(btn_reset)
        layout.add_widget(label)

        root = BoxLayout(orientation='vertical')
        rfl = FloatLayout()
        rfl.add_widget(wid)
        root.add_widget(rfl)
        root.add_widget(layout)


        return root


if __name__ == '__main__':

	StencilCanvasApp().run()
