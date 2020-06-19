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





# Load the model for inference 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

model = Net()
model.load_state_dict(torch.load('./MNIST_net.pth'))
model.eval() 



# Backgrounds of widgets 

Builder.load_string("""

<Button>:
    canvas.before:
        Color:
            rgba: .5, .5, .5, 1
        Line:
            width: 2
            rectangle: self.x, self.y, self.width, self.height

<StencilTestWidget>:
    text: 'test'
    canvas.before:
        Color:
            rgba: .5, .5, .5, 1
        Line:
            width: 2
            rectangle: self.x, self.y, self.width, self.height


       
""") 





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


class mnistAIApp(App):
 

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
    	transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    	img_array = transform(img_array)
    	img_array = img_array.unsqueeze(1) # add a channel dimension
    	output = model(img_array)
    	output = torch.argmax(output)
    	label.text = str(output.item())

      
    def build(self):
        wid = StencilTestWidget(size_hint=(None, None), size=[462.0, -390.0], pos=[160.0, 505.0])

        label = Label(text='Prediction:')

        btn_submit = Button(text='Submit')
        btn_submit.bind(on_press=partial(self.take, wid, label))

        btn_reset = Button(text='Clear')
        btn_reset.bind(on_press=partial(self.clear, label, wid))

        layout = BoxLayout(size_hint=(1, None), height=50)
        layout.add_widget(btn_submit)
        layout.add_widget(btn_reset)
        layout.add_widget(label)

        root = BoxLayout(orientation='vertical')
        rfl = FloatLayout()
        rfl.add_widget(wid)
        root.add_widget(rfl)
        root.add_widget(layout)


        return root


if __name__ == '__main__':

	mnistAIApp().run()
