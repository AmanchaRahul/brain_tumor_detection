from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        label='Select Brain Scan Image',
        help_text='Upload a brain MRI scan image for tumor detection'
    )
