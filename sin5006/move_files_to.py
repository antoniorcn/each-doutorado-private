from lightning_sdk import Studio

studio = Studio(name="scratch-studio", teamspace="Language-model", user="antoniorcarvalhon")
studio.upload_file('C:\git\dados\sin50006\casia-fasd.zip', 'sin500006')