from BlurOrNot import BlurorNot

image_paths = ['/image1.jpg', '/image2.jpg', '/image3.jpg', '/image4.jpg', '/image5.jpg']
for i, image_path in enumerate(image_paths):
    blurornot, prob_blur = BlurorNot('TestImages'+image_path)
    print('Image',i+1)
    if blurornot == 1:
        print('The image is blurred')
    else:
        print('The image is not blurred')
    print('The probability that the image is blurred is ',prob_blur)