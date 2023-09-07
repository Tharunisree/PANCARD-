# import numpy
# import pandas
# import cv2 as opencv
# image = opencv.imread(input("Input a File location: "))
# index = ["color","color_name","hex","R","G","B"]
# dataset = pandas.read_csv("dataset.csv", names=index, header=None)
# clickstate = False
# r=0
# g=0
# b=0
# position_x = 0
# position_y = 0
# def recognizer(R,G,B):
# minimum = 1000
# for i in range(len(dataset)):
# d = abs(R - int(dataset.loc[i,"R"])) + abs(G - int(dataset.loc[i, "G"])) + abs(B - int(dataset.loc[i, "B"]))
# if(d<=minimum):
# minimum = d
# cname = dataset.loc[i, "color_name"]
# return cname
# def clickhandler(event, x, y,f,p):
# if event == opencv.EVENT_LBUTTONDBLCLK:
# global b,g,r,position_x,position_y, clickstate
# clickstate = True
# position_x = x
# position_y = y
# b,g,r = image[y,x]
# b = int(b)
# g = int(g)
# r = int(r)
# opencv.namedWindow('Color Recognition Application')
# opencv.setMouseCallback('Color Recognition Application', clickhandler)
# while(1):
# opencv.imshow("Color Recognition Application",image)
# if (clickstate):
# #opencv.rectangle(image, startpoint, endpoint, color, thickness)-1. It fills entire rectangle
# opencv.rectangle(image,(20,20), (750,60), (b,g,r), -1)
# #Creating a text string to display color name and RGB value
# text = recognizer(r,g,b) + "  RGB("+str(r)+","+str(g)+","+str(b)+")"
# #opencv.putText() function will write data on image
# opencv.putText(image, text,(50,50),2,0.8,(255,255,255),2,opencv.LINE_AA)
# #For light colors we will choose black color
# if(r+g+b>=600):
# opencv.putText(image, text,(50,50),2,0.8,(0,0,0),2,opencv.LINE_AA)
# clickstate=False
# #Ends the function when Esc is clicked
# if opencv.waitKey(20) & 0xFF ==27:
# break
# opencv.destroyAllWindows()























# import numpy
# import pandas
# import cv2 as opencv

# image = opencv.imread(input("C:\\Users\\durga\\Downloads\\pan-card-tamp-main\\pan-card-tamp-main\\sample_data\\dp_pancard.jpg"))
# index = ["color", "color_name", "hex", "R", "G", "B"]
# dataset = pandas.read_csv("C:\\Users\\durga\\Downloads\\color-recognization-master\\color-recognization-master\\dataset.csv", names=index, header=None)
# clickstate = False
# r = 0
# g = 0
# b = 0
# position_x = 0
# position_y = 0

# def recognizer(R, G, B):
#     minimum = 1000
#     for i in range(len(dataset)):
#         d = abs(R - int(dataset.loc[i, "R"])) + abs(G - int(dataset.loc[i, "G"])) + abs(B - int(dataset.loc[i, "B"]))
#         if d <= minimum:
#             minimum = d
#             cname = dataset.loc[i, "color_name"]
#     return cname

# def clickhandler(event, x, y, f, p):
#     global b, g, r, position_x, position_y, clickstate
#     if event == opencv.EVENT_LBUTTONDBLCLK:
#         clickstate = True
#         position_x = x
#         position_y = y
#         b, g, r = image[y, x]
#         b = int(b)
#         g = int(g)
#         r = int(r)

# opencv.namedWindow('Color Recognition Application')
# opencv.setMouseCallback('Color Recognition Application', clickhandler)

# while True:
#     opencv.imshow("Color Recognition Application", image)

#     if clickstate:
#         opencv.rectangle(image, (20, 20), (750, 60), (b, g, r), -1)
#         text = recognizer(r, g, b) + "  RGB(" + str(r) + "," + str(g) + "," + str(b) + ")"
#         opencv.putText(image, text, (50, 50), 2, 0.8, (255, 255, 255), 2, opencv.LINE_AA)

#         if r + g + b >= 600:
#             opencv.putText(image, text, (50, 50), 2, 0.8, (0, 0, 0), 2, opencv.LINE_AA)

#         clickstate = False

#     if opencv.waitKey(20) & 0xFF == 27:
#         break

# opencv.destroyAllWindows()


















import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


import cv2
import extcolors

from colormap import rgb2hex


# input_name = '<photo location/name>'
# output_width = 900                   #set the output size
# img = Image.open(input_name)
# wpercent = (output_width/float(img.size[0]))
# hsize = int((float(img.size[1])*float(wpercent)))
# img = img.resize((output_width,hsize), Image.ANTIALIAS)

# #save
# resize_name = 'resize_' + input_name  #the resized image name
# img.save(resize_name)                 #output location can be specified before resize_name

# #read
# plt.figure(figsize=(9, 9))
# img_url = resize_name
# img = plt.imread(img_url)
# plt.imshow(img)
# plt.axis('off')
# plt.show()

# colors_x = extcolors.extract_from_path(img_url, tolerance = 12, limit = 12)
# colors_x

# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]
    
#     #convert RGB to HEX code
#     df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
#                           int(i.split(", ")[1]),
#                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]
    
#     df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
#     return df

# df_color = color_to_df(colors_x)
# df_color

# list_color = list(df_color['c_code'])
# list_precent = [int(i) for i in list(df_color['occurence'])]
# text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color,
#                                                                                list_precent)]
# fig, ax = plt.subplots(figsize=(90,90),dpi=10)
# wedges, text = ax.pie(list_precent,
#                       labels= text_c,
#                       labeldistance= 1.05,
#                       colors = list_color,
#                       textprops={'fontsize': 120, 'color':'black'}
#                      )
# plt.setp(wedges, width=0.3)

# #create space in the center
# plt.setp(wedges, width=0.36)

# ax.set_aspect("equal")
# fig.set_facecolor('white')
# plt.show()


# #create background color
# fig, ax = plt.subplots(figsize=(192,108),dpi=10)
# fig.set_facecolor('white')
# plt.savefig('bg.png')
# plt.close(fig)

# #create color palette
# bg = plt.imread('bg.png')
# fig = plt.figure(figsize=(90, 90), dpi = 10)
# ax = fig.add_subplot(1,1,1)

# x_posi, y_posi, y_posi2 = 320, 25, 25
# for c in list_color:
#     if  list_color.index(c) <= 5:
#         y_posi += 125
#         rect = patches.Rectangle((x_posi, y_posi), 290, 115, facecolor = c)
#         ax.add_patch(rect)
#         ax.text(x = x_posi+360, y = y_posi+80, s = c, fontdict={'fontsize': 150})
#     else:
#         y_posi2 += 125
#         rect = patches.Rectangle((x_posi + 800, y_posi2), 290, 115, facecolor = c)
#         ax.add_artist(rect)
#         ax.text(x = x_posi+1160, y = y_posi2+80, s = c, fontdict={'fontsize': 150})
        
# ax.axis('off')
# plt.imshow(bg)
# plt.tight_layout()



#####################################################   MAIN CODE       ##########################################################
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import matplotlib.image as mpimg

# from PIL import Image
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# import cv2
# import extcolors

# from colormap import rgb2hex


# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]
    
#     #convert RGB to HEX code
#     df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
#                           int(i.split(", ")[1]),
#                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]
    
#     df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
#     return df

# def exact_color(input_image, resize, tolerance, zoom):
#     #background
#     bg = 'bg.png'
#     fig, ax = plt.subplots(figsize=(192,108),dpi=10)
#     fig.set_facecolor('white')
#     plt.savefig(bg)
#     plt.close(fig)
    
#     #resize
#     output_width = resize
#     img = Image.open(input_image)
#     if img.size[0] >= resize:
#         wpercent = (output_width/float(img.size[0]))
#         hsize = int((float(img.size[1])*float(wpercent)))
#         img = img.resize((output_width,hsize), Image.ANTIALIAS)
#         resize_name = 'resize_'+ input_image
#         img.save(resize_name)
#     else:
#         resize_name = input_image
    
#     #crate dataframe
#     img_url = resize_name
#     colors_x = extcolors.extract_from_path(img_url, tolerance = tolerance, limit = 13)
#     df_color = color_to_df(colors_x)
    
#     #annotate text
#     list_color = list(df_color['c_code'])
#     list_precent = [int(i) for i in list(df_color['occurence'])]
#     text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color, list_precent)]
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160,120), dpi = 10)
    
#     #donut plot
#     wedges, text = ax1.pie(list_precent,
#                            labels= text_c,
#                            labeldistance= 1.05,
#                            colors = list_color,
#                            textprops={'fontsize': 150, 'color':'black'})
#     plt.setp(wedges, width=0.3)

#     #add image in the center of donut plot
#     img = mpimg.imread(resize_name)
#     imagebox = OffsetImage(img, zoom=zoom)
#     ab = AnnotationBbox(imagebox, (0, 0))
#     ax1.add_artist(ab)
    
#     #color palette
#     x_posi, y_posi, y_posi2 = 160, -170, -170
#     for c in list_color:
#         if list_color.index(c) <= 5:
#             y_posi += 180
#             rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor = c)
#             ax2.add_patch(rect)
#             ax2.text(x = x_posi+400, y = y_posi+100, s = c, fontdict={'fontsize': 190})
#         else:
#             y_posi2 += 180
#             rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor = c)
#             ax2.add_artist(rect)
#             ax2.text(x = x_posi+1400, y = y_posi2+100, s = c, fontdict={'fontsize': 190})

#     fig.set_facecolor('white')
#     ax2.axis('off')
#     bg = plt.imread('bg.png')
#     plt.imshow(bg)       
#     plt.tight_layout()
#     return plt.show()


# # exact_color('fruits.jpg', 900, 12, 2.5)

# # exact_color('flowers1.jpg', 900, 12, 2.5)
# # exact_color('pan_OC.jpg', 900, 12, 2.5)


# # exact_color('flowers2.jpg', 900, 24, 2.5)



# exact_color('dp_pancard.jpg', 900, 24, 2.5)










################################################ MAIN CODE END   ###########################################################################







# import cv2
# import numpy as np
# import extcolors
# import matplotlib.pyplot as plt
# from PIL import Image
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import matplotlib.image as mpimg
# import matplotlib.patches as patches
# from colormap import rgb2hex
# import pandas as pd

# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]

#     #convert RGB to HEX code
#     df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
#                           int(i.split(", ")[1]),
#                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

#     df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
#     return df

# def exact_color(input_image, resize, tolerance, zoom):
#     #background
#     bg = 'bg.png'
#     fig, ax = plt.subplots(figsize=(192,108),dpi=10)
#     fig.set_facecolor('white')
#     plt.savefig(bg)
#     plt.close(fig)

#     #resize
#     output_width = resize
#     img = Image.open(input_image)
#     if img.size[0] >= resize:
#         wpercent = (output_width/float(img.size[0]))
#         hsize = int((float(img.size[1])*float(wpercent)))
#         img = img.resize((output_width,hsize), Image.ANTIALIAS)
#         resize_name = 'resize_'+ input_image
#         img.save(resize_name)
#     else:
#         resize_name = input_image

#     #crate dataframe
#     img_url = resize_name
#     colors_x = extcolors.extract_from_path(img_url, tolerance=tolerance, limit=13)
#     df_color = color_to_df(colors_x)

#     #annotate text
#     list_color = list(df_color['c_code'])
#     list_precent = [int(i) for i in list(df_color['occurence'])]
#     text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color, list_precent)]
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160,120), dpi=10)

#     #donut plot
#     wedges, text = ax1.pie(list_precent,
#                            labels=text_c,
#                            labeldistance=1.05,
#                            colors=list_color,
#                            textprops={'fontsize': 150, 'color':'black'})
#     plt.setp(wedges, width=0.3)

#     #add image in the center of donut plot
#     img = mpimg.imread(resize_name)
#     imagebox = OffsetImage(img, zoom=zoom)
#     ab = AnnotationBbox(imagebox, (0, 0))
#     ax1.add_artist(ab)

#     #color palette
#     x_posi, y_posi, y_posi2 = 160, -170, -170
#     for c in list_color:
#         if list_color.index(c) <= 5:
#             y_posi += 180
#             rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
#             ax2.add_patch(rect)
#             ax2.text(x=x_posi+400, y=y_posi+100, s=c, fontdict={'fontsize': 190})
#         else:
#             y_posi2 += 180
#             rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
#             ax2.add_artist(rect)
#             ax2.text(x=x_posi+1400, y=y_posi2+100, s=c, fontdict={'fontsize': 190})

#     fig.set_facecolor('white')
#     ax2.axis('off')
#     bg = plt.imread('bg.png')
#     plt.imshow(bg)
#     plt.tight_layout()
#     return plt.show()


# # Function to capture frames from webcam
# def capture_frames():
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()

#         cv2.imshow('Webcam', frame)

#         # Press 'q' to exit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return frame


# # Save the captured frame as an image file
# def save_frame(frame, filename):
#     cv2.imwrite(filename, frame)


# # Call the functions to capture the frame and save it as an image
# frame = capture_frames()
# save_frame(frame, 'webcam_image.jpg')

# # Call the color extraction function with the captured image
# exact_color('webcam_image.jpg', 900, 12, 2.5)


#################################################### With Green Box  ################################################################

# import cv2
# import numpy as np
# import extcolors
# import matplotlib.pyplot as plt
# from PIL import Image
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import matplotlib.image as mpimg
# import matplotlib.patches as patches
# from colormap import rgb2hex
# import pandas as pd

# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]

#     #convert RGB to HEX code
#     df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
#                           int(i.split(", ")[1]),
#                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

#     df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
#     return df

# def exact_color(input_image, resize, tolerance, zoom):
#     #background
#     bg = 'bg.png'
#     fig, ax = plt.subplots(figsize=(192,108),dpi=10)
#     fig.set_facecolor('white')
#     plt.savefig(bg)
#     plt.close(fig)

#     #resize
#     output_width = resize
#     img = Image.open(input_image)
#     if img.size[0] >= resize:
#         wpercent = (output_width/float(img.size[0]))
#         hsize = int((float(img.size[1])*float(wpercent)))
#         img = img.resize((output_width,hsize), Image.ANTIALIAS)
#         resize_name = 'resize_'+ input_image
#         img.save(resize_name)
#     else:
#         resize_name = input_image

#     #create dataframe
#     img_url = resize_name
#     colors_x = extcolors.extract_from_path(img_url, tolerance=tolerance, limit=13)
#     df_color = color_to_df(colors_x)

#     #annotate text
#     list_color = list(df_color['c_code'])
#     list_precent = [int(i) for i in list(df_color['occurence'])]
#     text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color, list_precent)]
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160,120), dpi=10)

#     #donut plot
#     wedges, text = ax1.pie(list_precent,
#                            labels=text_c,
#                            labeldistance=1.05,
#                            colors=list_color,
#                            textprops={'fontsize': 150, 'color':'black'})
#     plt.setp(wedges, width=0.3)

#     #add image in the center of donut plot
#     img = mpimg.imread(resize_name)
#     imagebox = OffsetImage(img, zoom=zoom)
#     ab = AnnotationBbox(imagebox, (0, 0))
#     ax1.add_artist(ab)

#     #color palette
#     x_posi, y_posi, y_posi2 = 160, -170, -170
#     for c in list_color:
#         if list_color.index(c) <= 5:
#             y_posi += 180
#             rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
#             ax2.add_patch(rect)
#             ax2.text(x=x_posi+400, y=y_posi+100, s=c, fontdict={'fontsize': 190})
#         else:
#             y_posi2 += 180
#             rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
#             ax2.add_artist(rect)
#             ax2.text(x=x_posi+1400, y=y_posi2+100, s=c, fontdict={'fontsize': 190})

#     fig.set_facecolor('white')
#     ax2.axis('off')
#     bg = plt.imread('bg.png')
#     plt.imshow(bg)
#     plt.tight_layout()
#     return plt.show()


# # Function to capture frames from webcam
# def capture_frames():
#     cap = cv2.VideoCapture(0)

#     # Get the dimensions of the webcam frame
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Calculate the position of the rectangle box
#     rect_width = 250
#     rect_height = 350
#     rect_x = int((frame_width - rect_width) / 2)
#     rect_y = int((frame_height - rect_height) / 2)
#     rect_size = (rect_width, rect_height)

#     while True:
#         ret, frame = cap.read()

#         # Display the rectangle box
#         cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)
#         card_region = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]

#         cv2.imshow('Webcam', frame)

#         # Press 'q' to exit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return card_region


# # Save the captured card region as an image file
# def save_card_region(card_region, filename):
#     cv2.imwrite(filename, card_region)


# # Call the functions to capture the card region and save it as an image
# card_region = capture_frames()
# save_card_region(card_region, 'card_image.jpg')

# # Call the color extraction function with the captured card image
# exact_color('card_image.jpg', 900, 12, 2.5)


############################################## Modifications of green box ############################################################################


import cv2
import numpy as np
import extcolors
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import matplotlib.patches as patches
from colormap import rgb2hex
import pandas as pd

def color_to_df(input):
    colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
    df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
    df_percent = [float(i.split('), ')[1].replace(')','')) for i in colors_pre_list]

    # convert RGB to HEX code
    df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
                      int(i.split(", ")[1]),
                      int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

    total_percent = sum(df_percent)
    df_percent = [(p / total_percent) * 100 for p in df_percent]

    df = pd.DataFrame(zip(df_color_up, df_percent), columns=['c_code', 'percentage'])  # Corrected column name
    return df


# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]

#     #convert RGB to HEX code
#     df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
#                           int(i.split(", ")[1]),
#                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

#     df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
#     return df

def exact_color(input_image, resize, tolerance, zoom):
    #background
    bg = 'bg.png'
    fig, ax = plt.subplots(figsize=(192,108),dpi=10)
    fig.set_facecolor('white')
    plt.savefig(bg)
    plt.close(fig)

    #resize
    output_width = resize
    img = Image.open(input_image)
    if img.size[0] >= resize:
        wpercent = (output_width/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((output_width,hsize), Image.ANTIALIAS)
        resize_name = 'resize_'+ input_image
        img.save(resize_name)
    else:
        resize_name = input_image

    #create dataframe
    img_url = resize_name
    colors_x = extcolors.extract_from_path(img_url, tolerance=tolerance, limit=13)
    df_color = color_to_df(colors_x)

    #annotate text
    list_color = list(df_color['c_code'])
    list_precent = [int(i) for i in list(df_color['percentage'])]
    text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color, list_precent)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160,120), dpi=10)

    # print(text_c)

    #donut plot
    wedges, text = ax1.pie(list_precent,
                           labels=text_c,
                           labeldistance=1.05,
                           colors=list_color,
                           textprops={'fontsize': 150, 'color':'black'})
    plt.setp(wedges, width=0.3)

    #add image in the center of donut plot
    img = mpimg.imread(resize_name)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (0, 0))
    ax1.add_artist(ab)

    #color palette
    x_posi, y_posi, y_posi2 = 160, -170, -170
    for c in list_color:
        if list_color.index(c) <= 5:
            y_posi += 180
            rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
            ax2.add_patch(rect)
            ax2.text(x=x_posi+400, y=y_posi+100, s=c, fontdict={'fontsize': 190})
        else:
            y_posi2 += 180
            rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
            ax2.add_artist(rect)
            ax2.text(x=x_posi+1400, y=y_posi2+100, s=c, fontdict={'fontsize': 190})

    fig.set_facecolor('white')
    ax2.axis('off')
    bg = plt.imread('bg.png')
    plt.imshow(bg)
    plt.tight_layout()
    color_data = df_color.to_dict(orient='records')   ##
    plt.show()
    return color_data  ##
    

# Function to draw L shapes at the four corners
def draw_L_shapes(frame, rect_x, rect_y, rect_width, rect_height, l_size):
    thickness = 2
    color = (0, 255, 0)  # Green color

    # Top-left L shape
    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + l_size, rect_y + thickness), color, -1)
    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + thickness, rect_y + l_size), color, -1)
 
    # Top-right L shape
    cv2.rectangle(frame, (rect_x + rect_width - l_size, rect_y), (rect_x + rect_width, rect_y + thickness), color, -1)
    cv2.rectangle(frame, (rect_x + rect_width - thickness, rect_y), (rect_x + rect_width, rect_y + l_size), color, -1)

    # Bottom-left L shape
    cv2.rectangle(frame, (rect_x, rect_y + rect_height - thickness), (rect_x + l_size, rect_y + rect_height), color, -1)
    cv2.rectangle(frame, (rect_x, rect_y + rect_height - l_size), (rect_x + thickness, rect_y + rect_height), color, -1)

    # Bottom-right L shape
    cv2.rectangle(frame, (rect_x + rect_width - l_size, rect_y + rect_height - thickness), (rect_x + rect_width, rect_y + rect_height), color, -1)
    cv2.rectangle(frame, (rect_x + rect_width - thickness, rect_y + rect_height - l_size), (rect_x + rect_width, rect_y + rect_height), color, -1)

    return frame

# Function to capture frames from webcam
def capture_frames():
    cap = cv2.VideoCapture(0)

    # Get the dimensions of the webcam frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the position of the rectangle box
    rect_width = 400
    rect_height = 350
    rect_x = int((frame_width - rect_width) / 2)
    rect_y = int((frame_height - rect_height) / 2)
    rect_size = (rect_width, rect_height)

    # Set the L shape size for corners
    l_size = 30

    while True:
        ret, frame = cap.read()

        # Draw L shapes at the corners
        frame = draw_L_shapes(frame, rect_x, rect_y, rect_width, rect_height, l_size)

        # Display the webcam frame
        cv2.imshow('Webcam', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]

# Save the captured card region as an image file
def save_card_region(card_region, filename):
    cv2.imwrite(filename, card_region)


# Call the functions to capture the card region and save it as an image
card_region = capture_frames()
save_card_region(card_region, 'card_image.jpg')

# Call the color extraction function with the captured card image
exact_color('card_image.jpg', 900, 12, 2.5)



# import cv2
# import numpy as np
# import extcolors
# import matplotlib.pyplot as plt
# from PIL import Image
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import matplotlib.image as mpimg
# import matplotlib.patches as patches
# from colormap import rgb2hex
# import pandas as pd
# import os
# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]

#     #convert RGB to HEX code
#     df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
#                           int(i.split(", ")[1]),
#                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

#     df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
#     return df

# def exact_color(input_image, resize, tolerance, zoom):
#     #background
#     bg = 'bg.png'
#     fig, ax = plt.subplots(figsize=(192,108),dpi=10)
#     fig.set_facecolor('white')
#     plt.savefig(bg)
#     plt.close(fig)

#     #resize
#     output_width = resize
#     img = Image.open(input_image)
#     if img.size[0] >= resize:
#         wpercent = (output_width/float(img.size[0]))
#         hsize = int((float(img.size[1])*float(wpercent)))
#         img = img.resize((output_width,hsize), Image.ANTIALIAS)
#         resize_name = 'resize_'+ input_image
#         img.save(resize_name)
#     else:
#         resize_name = input_image

#     #create dataframe
#     img_url = resize_name
#     colors_x = extcolors.extract_from_path(img_url, tolerance=tolerance, limit=13)
#     df_color = color_to_df(colors_x)

#     #annotate text
#     list_color = list(df_color['c_code'])
#     list_precent = [int(i) for i in list(df_color['occurence'])]
#     text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color, list_precent)]
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160,120), dpi=10)

#     #donut plot
#     wedges, text = ax1.pie(list_precent,
#                            labels=text_c,
#                            labeldistance=1.05,
#                            colors=list_color,
#                            textprops={'fontsize': 150, 'color':'black'})
#     plt.setp(wedges, width=0.3)

#     #add image in the center of donut plot
#     img = mpimg.imread(resize_name)
#     imagebox = OffsetImage(img, zoom=zoom)
#     ab = AnnotationBbox(imagebox, (0, 0))
#     ax1.add_artist(ab)

#     #color palette
#     x_posi, y_posi, y_posi2 = 160, -170, -170
#     for c in list_color:
#         if list_color.index(c) <= 5:
#             y_posi += 180
#             rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
#             ax2.add_patch(rect)
#             ax2.text(x=x_posi+400, y=y_posi+100, s=c, fontdict={'fontsize': 190})
#         else:
#             y_posi2 += 180
#             rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
#             ax2.add_artist(rect)
#             ax2.text(x=x_posi+1400, y=y_posi2+100, s=c, fontdict={'fontsize': 190})

#     fig.set_facecolor('white')
#     ax2.axis('off')
#     bg = plt.imread('bg.png')
#     plt.imshow(bg)
#     plt.tight_layout()
#     return plt.show()


# # Function to draw L shapes at the four corners
# def draw_L_shapes(frame, rect_x, rect_y, rect_width, rect_height, l_size):
#     thickness = 2
#     color = (0, 255, 0)  # Green color

#     # Top-left L shape
#     cv2.rectangle(frame, (rect_x, rect_y), (rect_x + l_size, rect_y + thickness), color, -1)
#     cv2.rectangle(frame, (rect_x, rect_y), (rect_x + thickness, rect_y + l_size), color, -1)

#     # Top-right L shape
#     cv2.rectangle(frame, (rect_x + rect_width - l_size, rect_y), (rect_x + rect_width, rect_y + thickness), color, -1)
#     cv2.rectangle(frame, (rect_x + rect_width - thickness, rect_y), (rect_x + rect_width, rect_y + l_size), color, -1)

#     # Bottom-left L shape
#     cv2.rectangle(frame, (rect_x, rect_y + rect_height - thickness), (rect_x + l_size, rect_y + rect_height), color, -1)
#     cv2.rectangle(frame, (rect_x, rect_y + rect_height - l_size), (rect_x + thickness, rect_y + rect_height), color, -1)

#     # Bottom-right L shape
#     cv2.rectangle(frame, (rect_x + rect_width - l_size, rect_y + rect_height - thickness), (rect_x + rect_width, rect_y + rect_height), color, -1)
#     cv2.rectangle(frame, (rect_x + rect_width - thickness, rect_y + rect_height - l_size), (rect_x + rect_width, rect_y + rect_height), color, -1)

#     return frame

# # Function to capture frames from webcam
# def capture_frames():
#     cap = cv2.VideoCapture(0)

#     # Get the dimensions of the webcam frame
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Calculate the position of the rectangle box
#     rect_width = 250
#     rect_height = 350
#     rect_x = int((frame_width - rect_width) / 2)
#     rect_y = int((frame_height - rect_height) / 2)
#     rect_size = (rect_width, rect_height)

#     # Set the L shape size for corners
#     l_size = 30

#     while True:
#         ret, frame = cap.read()

#         # Draw L shapes at the corners
#         frame = draw_L_shapes(frame, rect_x, rect_y, rect_width, rect_height, l_size)

#         # Display the webcam frame
#         cv2.imshow('Webcam', frame)

#         # Press 'q' to exit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]

# # Save the captured card region as an image file
# def save_card_region(card_region, filename):
#     cv2.imwrite(filename, card_region)

# # Function to resize an image
# def resize_image(input_image, output_image, output_width):
#     img = Image.open(input_image)
#     if img.size[0] >= output_width:
#         wpercent = (output_width / float(img.size[0]))
#         hsize = int((float(img.size[1]) * float(wpercent)))
#         img = img.resize((output_width, hsize), Image.ANTIALIAS)
#         img.save(output_image)
# # Function to extract color from a manually uploaded image
# def extract_colors_from_uploaded_image(upload_path, resize, tolerance, zoom):
#     # Extract the original filename without the path
#     original_filename = os.path.basename(upload_path)

#     # Create the resized filename with the desired format
#     resized_filename = 'resized_' + original_filename

#     # Resize the image and save it with the resized filename
#     resize_image(upload_path, resized_filename, resize)
#     exact_color(resized_filename, resize, tolerance, zoom)



# # Main function to process both webcam and uploaded images
# def main():
#     # Capture card region from webcam
#     card_region_webcam = capture_frames()

#     # Save the captured card region as an image file
#     save_card_region(card_region_webcam, 'card_image_webcam.jpg')

#     # Extract colors from webcam-captured image
#     exact_color('card_image_webcam.jpg', 900, 12, 2.5)
#     webcam_image = mpimg.imread('card_image_webcam.jpg')

#     # Manually upload an image (replace 'path_to_uploaded_image' with the actual path)
#     uploaded_image_path = 'C:\\Users\\durga\\Downloads\\Color_extraction\\dp_pancard.jpg'

#     # Extract colors from uploaded image
#     extract_colors_from_uploaded_image(uploaded_image_path, 900, 12, 2.5)
#     uploaded_image = mpimg.imread('resized_dp_pancard.jpg')  # Change the filename accordingly

#     # Resize both images to the same height (adjust the width as needed)
#     desired_height = 600
#     webcam_image_resized = cv2.resize(webcam_image, (int(desired_height * webcam_image.shape[1] / webcam_image.shape[0]), desired_height))
#     uploaded_image_resized = cv2.resize(uploaded_image, (int(desired_height * uploaded_image.shape[1] / uploaded_image.shape[0]), desired_height))

#     # Concatenate the resized images horizontally
#     concatenated_images = cv2.hconcat([webcam_image_resized, uploaded_image_resized])

#     # Display the concatenated image using OpenCV
#     cv2.imshow('Color Extraction Results', concatenated_images)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


# # Main function to process both webcam and uploaded images
# def main():
#     # Capture card region from webcam
#     card_region_webcam = capture_frames()

#     # Save the captured card region as an image file
#     save_card_region(card_region_webcam, 'card_image_webcam.jpg')

#     # Extract colors from webcam-captured image
#     exact_color('card_image_webcam.jpg', 900, 12, 2.5)

#     # Manually upload an image (replace 'path_to_uploaded_image' with the actual path)
#     uploaded_image_path = 'C:\\Users\\durga\\Downloads\\Color_extraction\\dp_pancard.jpg'

#     # Extract colors from uploaded image
#     extract_colors_from_uploaded_image(uploaded_image_path, 900, 12, 2.5)

# ... (previous code remains unchanged)


# # Main function to process both webcam and uploaded images
# def main():
#     # Capture card region from webcam
#     card_region_webcam = capture_frames()

#     # Save the captured card region as an image file
#     save_card_region(card_region_webcam, 'card_image_webcam.jpg')

#     # Extract colors from webcam-captured image
#     webcam_colors = exact_color('card_image_webcam.jpg', 900, 12, 2.5)

#     # Manually upload an image (replace 'path_to_uploaded_image' with the actual path)
#     uploaded_image_path = 'C:\\Users\\durga\\Downloads\\Color_extraction\\dp_pancard.jpg'

#     # Extract colors from uploaded image
#     uploaded_colors = extract_colors_from_uploaded_image(uploaded_image_path, 900, 12, 2.5)

#     # Print color extraction results side by side
#     print_color_extraction_results(webcam_colors, uploaded_colors)# Call the main function

# # Function to print color extraction results
# def print_color_extraction_results(colors1, colors2):
#     # Combine and print color extraction results side by side
#     max_color_count = max(len(colors1), len(colors2))
#     print("Webcam Colors".center(40) + "|" + "Uploaded Image Colors".center(40))
#     print("-" * 80)
#     for i in range(max_color_count):
#         color1 = colors1[i] if i < len(colors1) else ("", "")
#         color2 = colors2[i] if i < len(colors2) else ("", "")
#         print(f"{color1[0]}: {color1[1]}".ljust(40) + "|" + f"{color2[0]}: {color2[1]}".ljust(40))

# if __name__ == "__main__":
#     main()
###################################################################




# import cv2
# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# from PIL import Image
# import matplotlib.image as mpimg
# import matplotlib.patches as patches
# from colormap import rgb2hex
# import pandas as pd

# import cv2
# import numpy as np
# import extcolors
# import matplotlib.pyplot as plt
# from PIL import Image
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import matplotlib.image as mpimg
# import matplotlib.patches as patches
# from colormap import rgb2hex
# import pandas as pd

# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]

#     #convert RGB to HEX code
#     df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
#                           int(i.split(", ")[1]),
#                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

#     df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
#     return df

# def exact_color(input_image, resize, tolerance, zoom):
#     #background
#     bg = 'bg.png'
#     fig, ax = plt.subplots(figsize=(192,108),dpi=10)
#     fig.set_facecolor('white')
#     plt.savefig(bg)
#     plt.close(fig)

#     #resize
#     output_width = resize
#     img = Image.open(input_image)
#     if img.size[0] >= resize:
#         wpercent = (output_width/float(img.size[0]))
#         hsize = int((float(img.size[1])*float(wpercent)))
#         img = img.resize((output_width,hsize), Image.ANTIALIAS)
#         resize_name = 'resize_'+ input_image
#         img.save(resize_name)
#     else:
#         resize_name = input_image

#     #create dataframe
#     img_url = resize_name
#     colors_x = extcolors.extract_from_path(img_url, tolerance=tolerance, limit=13)
#     df_color = color_to_df(colors_x)

#     #annotate text
#     list_color = list(df_color['c_code'])
#     list_precent = [int(i) for i in list(df_color['occurence'])]
#     text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color, list_precent)]
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160,120), dpi=10)

#     #donut plot
#     wedges, text = ax1.pie(list_precent,
#                            labels=text_c,
#                            labeldistance=1.05,
#                            colors=list_color,
#                            textprops={'fontsize': 150, 'color':'black'})
#     plt.setp(wedges, width=0.3)

#     #add image in the center of donut plot
#     img = mpimg.imread(resize_name)
#     imagebox = OffsetImage(img, zoom=zoom)
#     ab = AnnotationBbox(imagebox, (0, 0))
#     ax1.add_artist(ab)

#     #color palette
#     x_posi, y_posi, y_posi2 = 160, -170, -170
#     for c in list_color:
#         if list_color.index(c) <= 5:
#             y_posi += 180
#             rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
#             ax2.add_patch(rect)
#             ax2.text(x=x_posi+400, y=y_posi+100, s=c, fontdict={'fontsize': 190})
#         else:
#             y_posi2 += 180
#             rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
#             ax2.add_artist(rect)
#             ax2.text(x=x_posi+1400, y=y_posi2+100, s=c, fontdict={'fontsize': 190})

#     fig.set_facecolor('white')
#     ax2.axis('off')
#     bg = plt.imread('bg.png')
#     plt.imshow(bg)
#     plt.tight_layout()
#     return plt.show()


# # Function to capture an image inside a frame from the webcam
# def capture_card_region():
#     cap = cv2.VideoCapture(0)

#     # Get the dimensions of the webcam frame
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Calculate the position of the rectangle box
#     rect_width = 250
#     rect_height = 350
#     rect_x = int((frame_width - rect_width) / 2)
#     rect_y = int((frame_height - rect_height) / 2)
#     rect_size = (rect_width, rect_height)

#     while True:
#         ret, frame = cap.read()

#         # Display the rectangle box
#         cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)
#         cv2.imshow('Webcam', frame)

#         # Press 'c' to capture the card region and 'q' to exit the loop
#         key = cv2.waitKey(1)
#         if key == ord('c'):
#             card_region = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]
#             break
#         elif key == ord('q'):
#             card_region = None
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return card_region

# # Save the captured card region as an image file
# def save_card_region(card_region, filename):
#     cv2.imwrite(filename, card_region)

# def main():
#     # Capture the card region from the webcam and save it as an image
#     card_region = capture_card_region()
#     if card_region is not None:
#         save_card_region(card_region, 'card_image.jpg')

#         # Call the color extraction function with the captured card image
#         exact_color('card_image.jpg', 900, 12, 2.5)

# if __name__ == "__main__":
#     main()







# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]

#     # Convert RGB to HEX code
#     df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
#                           int(i.split(", ")[1]),
#                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

#     df = pd.DataFrame(zip(df_color_up, df_percent), columns=['c_code', 'occurence'])
#     return df

# def extract_colors(input_image, resize, num_colors, zoom):
#     # Resize the input image
#     output_width = resize
#     img = Image.open(input_image)
#     if img.size[0] >= resize:
#         wpercent = (output_width / float(img.size[0]))
#         hsize = int((float(img.size[1]) * float(wpercent)))
#         img = img.resize((output_width, hsize), Image.ANTIALIAS)
#         resize_name = 'resize_' + input_image
#         img.save(resize_name)
#     else:
#         resize_name = input_image

#     # Create a mask to remove the background (in this case, green)
#     img_cv2 = cv2.imread(resize_name)
#     hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
#     lower_green = np.array([40, 40, 40])
#     upper_green = np.array([80, 255, 255])
#     mask = cv2.inRange(hsv, lower_green, upper_green)
#     masked_image = cv2.bitwise_and(img_cv2, img_cv2, mask=mask)

#     # Convert the masked image back to PIL Image format
#     img_no_bg = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))

#     # Convert image to numpy array
#     img_array = np.array(img_no_bg)

#     # Reshape the image to be a list of RGB pixels
#     pixels = img_array.reshape(-1, 3)

#     # Apply KMeans clustering to get dominant colors
#     kmeans = KMeans(n_clusters=num_colors)
#     kmeans.fit(pixels)

#     # Get the cluster centers (representative colors)
#     colors = kmeans.cluster_centers_
#     colors = colors.astype(int)

#     # Convert RGB to HEX code
#     list_color = ['#' + rgb2hex(color[0], color[1], color[2]) for color in colors]

#     # Annotate text and create the color palette
#     list_percent = [(len(np.where(kmeans.labels_ == i)[0]) / len(kmeans.labels_)) * 100 for i in range(num_colors)]
#     text_c = [c + ' ' + str(round(p, 1)) + '%' for c, p in zip(list_color, list_percent)]
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160, 120), dpi=10)

#     # Donut plot
#     wedges, text = ax1.pie(list_percent,
#                            labels=text_c,
#                            labeldistance=1.05,
#                            colors=list_color,
#                            textprops={'fontsize': 150, 'color': 'black'})
#     plt.setp(wedges, width=0.3)

#     # Add image in the center of the donut plot
#     img = mpimg.imread(resize_name)
#     imagebox = OffsetImage(img, zoom=zoom)
#     ab = AnnotationBbox(imagebox, (0, 0))
#     ax1.add_artist(ab)

#     # Color palette
#     x_posi, y_posi, y_posi2 = 160, -170, -170
#     for c in list_color:
#         if list_color.index(c) <= 5:
#             y_posi += 180
#             rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
#             ax2.add_patch(rect)
#             ax2.text(x=x_posi+400, y=y_posi+100, s=c, fontdict={'fontsize': 190})
#         else:
#             y_posi2 += 180
#             rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
#             ax2.add_artist(rect)
#             ax2.text(x=x_posi+1400, y=y_posi2+100, s=c, fontdict={'fontsize': 190})

#     fig.set_facecolor('white')
#     ax2.axis('off')
#     bg = plt.imread('bg.png')
#     plt.imshow(bg)
#     plt.tight_layout()
#     return plt.show()

# # Function to capture frames from webcam
# def capture_frames():
#     cap = cv2.VideoCapture(0)

#     # Get the dimensions of the webcam frame
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Calculate the position of the rectangle box
#     rect_width = 250
#     rect_height = 350
#     rect_x = int((frame_width - rect_width) / 2)
#     rect_y = int((frame_height - rect_height) / 2)
#     rect_size = (rect_width, rect_height)

#     while True:
#         ret, frame = cap.read()

#         # Display the rectangle box
#         cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)
#         card_region = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]

#         cv2.imshow('Webcam', frame)

#         # Press 'q' to exit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return card_region

# # Save the captured card region as an image file
# def save_card_region(card_region, filename):
#     cv2.imwrite(filename, card_region)

# # Call the functions to capture the card region and save it as an image
# card_region = capture_frames()
# save_card_region(card_region, 'card_image.jpg')

# # Call the modified color extraction function with the captured card image
# extract_colors('card_image.jpg', 900, 13, 2.5)


# import cv2
# import numpy as np
# import extcolors
# import matplotlib.pyplot as plt
# from PIL import Image
# import matplotlib.image as mpimg
# import matplotlib.patches as patches
# from colormap import rgb2hex
# import pandas as pd

# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]

#     # Convert RGB to HEX code
#     df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
#                           int(i.split(", ")[1]),
#                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

#     df = pd.DataFrame(zip(df_color_up, df_percent), columns=['c_code', 'occurence'])
#     return df

# def extract_colors(input_image, resize, tolerance, zoom):
#     # Resize the input image
#     output_width = resize
#     img = Image.open(input_image)
#     if img.size[0] >= resize:
#         wpercent = (output_width / float(img.size[0]))
#         hsize = int((float(img.size[1]) * float(wpercent)))
#         img = img.resize((output_width, hsize), Image.ANTIALIAS)
#         resize_name = 'resize_' + input_image
#         img.save(resize_name)
#     else:
#         resize_name = input_image

#     # Create a mask to remove the background (in this case, green)
#     img_cv2 = cv2.imread(resize_name)
#     hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
#     lower_green = np.array([40, 40, 40])
#     upper_green = np.array([80, 255, 255])
#     mask = cv2.inRange(hsv, lower_green, upper_green)
#     masked_image = cv2.bitwise_and(img_cv2, img_cv2, mask=mask)

#     # Convert the masked image back to PIL Image format
#     img_no_bg = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))

#     # Create dataframe with colors extracted from the image (excluding background)
#     colors_x = extcolors.extract(img_no_bg, tolerance=tolerance, limit=13)
#     df_color = color_to_df(colors_x)

#     # Annotate text and create the color palette
#     list_color = list(df_color['c_code'])
#     list_precent = [int(i) for i in list(df_color['occurence'])]
#     text_c = [c + ' ' + str(round(p*100/sum(list_precent), 1)) + '%' for c, p in zip(list_color, list_precent)]
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160, 120), dpi=10)

#     # Donut plot
#     wedges, text = ax1.pie(list_precent,
#                            labels=text_c,
#                            labeldistance=1.05,
#                            colors=list_color,
#                            textprops={'fontsize': 150, 'color': 'black'})
#     plt.setp(wedges, width=0.3)

#     # Add image in the center of the donut plot
#     img = mpimg.imread(resize_name)
#     imagebox = OffsetImage(img, zoom=zoom)
#     ab = AnnotationBbox(imagebox, (0, 0))
#     ax1.add_artist(ab)

#     # Color palette
#     x_posi, y_posi, y_posi2 = 160, -170, -170
#     for c in list_color:
#         if list_color.index(c) <= 5:
#             y_posi += 180
#             rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
#             ax2.add_patch(rect)
#             ax2.text(x=x_posi+400, y=y_posi+100, s=c, fontdict={'fontsize': 190})
#         else:
#             y_posi2 += 180
#             rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
#             ax2.add_artist(rect)
#             ax2.text(x=x_posi+1400, y=y_posi2+100, s=c, fontdict={'fontsize': 190})

#     fig.set_facecolor('white')
#     ax2.axis('off')
#     bg = plt.imread('bg.png')
#     plt.imshow(bg)
#     plt.tight_layout()
#     return plt.show()


# # Function to capture frames from webcam
# def capture_frames():
#     cap = cv2.VideoCapture(0)

#     # Get the dimensions of the webcam frame
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Calculate the position of the rectangle box
#     rect_width = 250
#     rect_height = 350
#     rect_x = int((frame_width - rect_width) / 2)
#     rect_y = int((frame_height - rect_height) / 2)
#     rect_size = (rect_width, rect_height)

#     while True:
#         ret, frame = cap.read()

#         # Display the rectangle box
#         cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)
#         card_region = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]

#         cv2.imshow('Webcam', frame)

#         # Press 'q' to exit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return card_region


# # Save the captured card region as an image file
# def save_card_region(card_region, filename):
#     cv2.imwrite(filename, card_region)


# # Call the functions to capture the card region and save it as an image
# card_region = capture_frames()
# save_card_region(card_region, 'card_image.jpg')

# # Call the modified color extraction function with the captured card image
# extract_colors('card_image.jpg', 900, 12, 2.5)



# import cv2
# import numpy as np
# import extcolors
# import matplotlib.pyplot as plt
# from PIL import Image
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import matplotlib.image as mpimg
# import matplotlib.patches as patches
# from colormap import rgb2hex
# import pandas as pd

# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]

#     #convert RGB to HEX code
#     df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
#                           int(i.split(", ")[1]),
#                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

#     df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
#     return df

# def exact_color(card_region, resize, tolerance, zoom):
#     #background
#     bg = 'bg.png'
#     fig, ax = plt.subplots(figsize=(192,108),dpi=10)
#     fig.set_facecolor('white')
#     plt.savefig(bg)
#     plt.close(fig)

#     #resize
#     output_width = resize
#     img = Image.fromarray(card_region)
#     if img.size[0] >= resize:
#         wpercent = (output_width/float(img.size[0]))
#         hsize = int((float(img.size[1])*float(wpercent)))
#         img = img.resize((output_width,hsize), Image.ANTIALIAS)
#         resize_name = 'resize_card_region.jpg'
#         img.save(resize_name)
#     else:
#         resize_name = 'card_region.jpg'

#     #create dataframe
#     img_url = resize_name
#     colors_x = extcolors.extract_from_path(img_url, tolerance=tolerance, limit=13)
#     df_color = color_to_df(colors_x)

#     #annotate text
#     list_color = list(df_color['c_code'])
#     list_precent = [int(i) for i in list(df_color['occurence'])]
#     text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color, list_precent)]
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160,120), dpi=10)

#     #donut plot
#     wedges, text = ax1.pie(list_precent,
#                            labels=text_c,
#                            labeldistance=1.05,
#                            colors=list_color,
#                            textprops={'fontsize': 150, 'color':'black'})
#     plt.setp(wedges, width=0.3)

#     #add image in the center of donut plot
#     img = mpimg.imread(resize_name)
#     imagebox = OffsetImage(img, zoom=zoom)
#     ab = AnnotationBbox(imagebox, (0, 0))
#     ax1.add_artist(ab)

#     #color palette
#     x_posi, y_posi, y_posi2 = 160, -170, -170
#     for c in list_color:
#         if list_color.index(c) <= 5:
#             y_posi += 180
#             rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
#             ax2.add_patch(rect)
#             ax2.text(x=x_posi+400, y=y_posi+100, s=c, fontdict={'fontsize': 190})
#         else:
#             y_posi2 += 180
#             rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
#             ax2.add_artist(rect)
#             ax2.text(x=x_posi+1400, y=y_posi2+100, s=c, fontdict={'fontsize': 190})

#     fig.set_facecolor('white')
#     ax2.axis('off')
#     bg = plt.imread('bg.png')
#     plt.imshow(bg)
#     plt.tight_layout()
#     plt.show()


# def capture_frames():
#     cap = cv2.VideoCapture(0)

#     # Get the dimensions of the webcam frame
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Calculate the position of the rectangle box
#     rect_width = 250
#     rect_height = 350
#     rect_x = int((frame_width - rect_width) / 2)
#     rect_y = int((frame_height - rect_height) / 2)
#     rect_size = (rect_width, rect_height)

#     while True:
#         ret, frame = cap.read()

#         # Display the rectangle box
#         cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)
#         card_region = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]

#         cv2.imshow('Webcam', frame)

#         # Press 'q' to extract the colors from the captured card region
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             exact_color(card_region, 900, 12, 2.5)
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# # Call the function to capture the card region from the webcam and extract colors
# capture_frames()


# import cv2
# import numpy as np
# import extcolors
# import matplotlib.pyplot as plt
# from PIL import Image
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import matplotlib.image as mpimg
# import matplotlib.patches as patches
# from colormap import rgb2hex
# import pandas as pd
# import os

# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]

#     #convert RGB to HEX code
#     df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
#                           int(i.split(", ")[1]),
#                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

#     df = pd.DataFrame(zip(df_color_up, df_percent), columns=['c_code', 'occurrence'])
#     return df

# def exact_color(card_region, resize, tolerance, zoom):
#     # Convert the OpenCV image (BGR) to a PIL image (RGB)
#     pil_image = Image.fromarray(cv2.cvtColor(card_region, cv2.COLOR_BGR2RGB))

#     # Save the PIL image to a temporary file
#     tmp_filename = 'tmp_card_image.jpg'
#     pil_image.save(tmp_filename)

#     # Resize
#     output_width = resize
#     if pil_image.size[0] >= resize:
#         wpercent = (output_width / float(pil_image.size[0]))
#         hsize = int((float(pil_image.size[1]) * float(wpercent)))
#         pil_image = pil_image.resize((output_width, hsize), Image.ANTIALIAS)

#     # Create dataframe
#     colors_x = extcolors.extract_from_path(tmp_filename, tolerance=tolerance, limit=13)
#     df_color = color_to_df(colors_x)

#     # Annotate text
#     list_color = list(df_color['c_code'])
#     list_precent = [int(i) for i in list(df_color['occurrence'])]
#     text_c = [c + ' ' + str(round(p * 100 / sum(list_precent), 1)) + '%' for c, p in zip(list_color, list_precent)]
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160, 120), dpi=10)

#     # Donut plot
#     wedges, text = ax1.pie(list_precent,
#                            labels=text_c,
#                            labeldistance=1.05,
#                            colors=list_color,
#                            textprops={'fontsize': 150, 'color': 'black'})
#     plt.setp(wedges, width=0.3)

#     # Add image in the center of the donut plot
#     img = mpimg.imread(tmp_filename)
#     imagebox = OffsetImage(img, zoom=zoom)
#     ab = AnnotationBbox(imagebox, (0, 0))
#     ax1.add_artist(ab)

#     # Color palette
#     x_posi, y_posi, y_posi2 = 160, -170, -170
#     for c in list_color:
#         if list_color.index(c) <= 5:
#             y_posi += 180
#             rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
#             ax2.add_patch(rect)
#             ax2.text(x=x_posi + 400, y=y_posi + 100, s=c, fontdict={'fontsize': 190})
#         else:
#             y_posi2 += 180
#             rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
#             ax2.add_artist(rect)
#             ax2.text(x=x_posi + 1400, y=y_posi2 + 100, s=c, fontdict={'fontsize': 190})

#     fig.set_facecolor('white')
#     ax2.axis('off')
#     bg = plt.imread('bg.png')
#     plt.imshow(bg)
#     plt.tight_layout()
#     plt.show()

#     # Remove the temporary file
#     os.remove(tmp_filename)

# # Function to capture frames from webcam
# def capture_frames():
#     cap = cv2.VideoCapture(0)

#     # Get the dimensions of the webcam frame
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Calculate the position of the rectangle box
#     rect_width = 250
#     rect_height = 350
#     rect_x = int((frame_width - rect_width) / 2)
#     rect_y = int((frame_height - rect_height) / 2)
#     rect_size = (rect_width, rect_height)

#     while True:
#         ret, frame = cap.read()

#         # Display the rectangle box
#         cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)
#         card_region = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]

#         cv2.imshow('Webcam', frame)

#         # Press 'q' to exit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return card_region
# # ... Previous code ...

# # Save the captured card region as a PIL image file
# def save_card_region(card_region, filename):
#     pil_image = Image.fromarray(cv2.cvtColor(card_region, cv2.COLOR_BGR2RGB))
#     pil_image.save(filename)

# # Call the functions to capture the card region and save it as an image
# card_region = capture_frames()
# save_card_region(card_region, 'card_image.jpg')

# # Call the color extraction function with the captured card image region
# exact_color('card_image.jpg', 900, 12, 2.5)


# import cv2
# import numpy as np
# import extcolors
# import matplotlib.pyplot as plt
# from PIL import Image
# import matplotlib.image as mpimg
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import matplotlib.patches as patches
# from colormap import rgb2hex
# import pandas as pd

# # ... (Keep the existing functions color_to_df and exact_color as they are)

# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]

#     #convert RGB to HEX code
#     df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
#                           int(i.split(", ")[1]),
#                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

#     df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
#     return df

# def exact_color(input_image, resize, tolerance, zoom):
#     #background
#     bg = 'bg.png'
#     fig, ax = plt.subplots(figsize=(192,108),dpi=10)
#     fig.set_facecolor('white')
#     plt.savefig(bg)
#     plt.close(fig)

#     #resize
#     output_width = resize
#     img = Image.open(input_image)
#     if img.size[0] >= resize:
#         wpercent = (output_width/float(img.size[0]))
#         hsize = int((float(img.size[1])*float(wpercent)))
#         img = img.resize((output_width,hsize), Image.ANTIALIAS)
#         resize_name = 'resize_'+ input_image
#         img.save(resize_name)
#     else:
#         resize_name = input_image

#     #create dataframe
#     img_url = resize_name
#     colors_x = extcolors.extract_from_path(img_url, tolerance=tolerance, limit=13)
#     df_color = color_to_df(colors_x)

#     #annotate text
#     list_color = list(df_color['c_code'])
#     list_precent = [int(i) for i in list(df_color['occurence'])]
#     text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color, list_precent)]
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160,120), dpi=10)

#     #donut plot
#     wedges, text = ax1.pie(list_precent,
#                            labels=text_c,
#                            labeldistance=1.05,
#                            colors=list_color,
#                            textprops={'fontsize': 150, 'color':'black'})
#     plt.setp(wedges, width=0.3)

#     #add image in the center of donut plot
#     img = mpimg.imread(resize_name)
#     imagebox = OffsetImage(img, zoom=zoom)
#     ab = AnnotationBbox(imagebox, (0, 0))
#     ax1.add_artist(ab)

#     #color palette
#     x_posi, y_posi, y_posi2 = 160, -170, -170
#     for c in list_color:
#         if list_color.index(c) <= 5:
#             y_posi += 180
#             rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
#             ax2.add_patch(rect)
#             ax2.text(x=x_posi+400, y=y_posi+100, s=c, fontdict={'fontsize': 190})
#         else:
#             y_posi2 += 180
#             rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
#             ax2.add_artist(rect)
#             ax2.text(x=x_posi+1400, y=y_posi2+100, s=c, fontdict={'fontsize': 190})

#     fig.set_facecolor('white')
#     ax2.axis('off')
#     bg = plt.imread('bg.png')
#     plt.imshow(bg)
#     plt.tight_layout()
#     return plt.show()


# # Function to capture frames from webcam
# def capture_frames():
#     cap = cv2.VideoCapture(0)

#     # Get the dimensions of the webcam frame
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Calculate the position of the rectangle box
#     rect_width = 250
#     rect_height = 350
#     rect_x = int((frame_width - rect_width) / 2)
#     rect_y = int((frame_height - rect_height) / 2)
#     rect_size = (rect_width, rect_height)

#     while True:
#         ret, frame = cap.read()

#         # Display the rectangle box
#         cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)

#         cv2.imshow('Webcam', frame)

#         # Press 'q' to extract colors from the captured region
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             card_region = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]
#             save_card_region(card_region, 'card_image.jpg')
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return 'card_image.jpg'


# # Save the captured card region as an image file
# def save_card_region(card_region, filename):
#     cv2.imwrite(filename, card_region)


# # Call the functions to capture the card region and save it as an image
# image_path = capture_frames()

# # Call the color extraction function with the captured card image
# exact_color(image_path, 900, 12, 2.5)






























# import cv2
# import numpy as np
# import extcolors
# import matplotlib.pyplot as plt
# from PIL import Image
# import matplotlib.image as mpimg
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import matplotlib.patches as patches
# from colormap import rgb2hex
# import pandas as pd

# # ... (Keep the existing functions color_to_df and exact_color as they are)

# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]

#     #convert RGB to HEX code
#     df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
#                           int(i.split(", ")[1]),
#                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

#     df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
#     return df

# def exact_color(input_image, resize, tolerance, zoom):
#     #background
#     bg = 'bg.png'
#     fig, ax = plt.subplots(figsize=(192,108),dpi=10)
#     fig.set_facecolor('white')
#     plt.savefig(bg)
#     plt.close(fig)

#     #resize
#     output_width = resize
#     img = Image.open(input_image)
#     if img.size[0] >= resize:
#         wpercent = (output_width/float(img.size[0]))
#         hsize = int((float(img.size[1])*float(wpercent)))
#         img = img.resize((output_width,hsize), Image.ANTIALIAS)
#         resize_name = 'resize_'+ input_image
#         img.save(resize_name)
#     else:
#         resize_name = input_image

#     #create dataframe
#     img_url = resize_name
#     colors_x = extcolors.extract_from_path(img_url, tolerance=tolerance, limit=13)
#     df_color = color_to_df(colors_x)

#     #annotate text
#     list_color = list(df_color['c_code'])
#     list_precent = [int(i) for i in list(df_color['occurence'])]
#     text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color, list_precent)]
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160,120), dpi=10)

#     #donut plot
#     wedges, text = ax1.pie(list_precent,
#                            labels=text_c,
#                            labeldistance=1.05,
#                            colors=list_color,
#                            textprops={'fontsize': 150, 'color':'black'})
#     plt.setp(wedges, width=0.3)

#     #add image in the center of donut plot
#     img = mpimg.imread(resize_name)
#     imagebox = OffsetImage(img, zoom=zoom)
#     ab = AnnotationBbox(imagebox, (0, 0))
#     ax1.add_artist(ab)

#     #color palette
#     x_posi, y_posi, y_posi2 = 160, -170, -170
#     for c in list_color:
#         if list_color.index(c) <= 5:
#             y_posi += 180
#             rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
#             ax2.add_patch(rect)
#             ax2.text(x=x_posi+400, y=y_posi+100, s=c, fontdict={'fontsize': 190})
#         else:
#             y_posi2 += 180
#             rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
#             ax2.add_artist(rect)
#             ax2.text(x=x_posi+1400, y=y_posi2+100, s=c, fontdict={'fontsize': 190})

#     fig.set_facecolor('white')
#     ax2.axis('off')
#     bg = plt.imread('bg.png')
#     plt.imshow(bg)
#     plt.tight_layout()
#     return plt.show()


# # Function to capture frames from webcam
# def capture_frames():
#     cap = cv2.VideoCapture(0)

#     # Get the dimensions of the webcam frame
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Calculate the position of the rectangle box
#     rect_width = 250
#     rect_height = 350
#     rect_x = int((frame_width - rect_width) / 2)
#     rect_y = int((frame_height - rect_height) / 2)
#     rect_size = (rect_width, rect_height)

#     while True:
#         ret, frame = cap.read()

#         # Display the rectangle box
#         cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)

#         cv2.imshow('Webcam', frame)

#         # Press 'q' to extract colors from the captured region
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             card_region = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]
#             save_card_region(card_region, 'card_image.jpg')
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return 'card_image.jpg'


# # Save the captured card region as an image file
# def save_card_region(card_region, filename):
#     cv2.imwrite(filename, card_region)


# # Call the functions to capture the card region and save it as an image
# image_path = capture_frames()

# # Call the color extraction function with the captured card image
# exact_color(image_path, 900, 12, 2.5)










# import cv2
# import numpy as np
# import extcolors
# import matplotlib.pyplot as plt
# from PIL import Image
# import matplotlib.image as mpimg
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import matplotlib.patches as patches
# from colormap import rgb2hex
# import pandas as pd

# # ... (Keep the existing functions color_to_df and exact_color as they are)

# import cv2
# import numpy as np
# import extcolors
# import matplotlib.pyplot as plt
# from PIL import Image
# import matplotlib.image as mpimg
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import matplotlib.patches as patches
# from colormap import rgb2hex
# import pandas as pd

# # ... (Keep the existing functions color_to_df and exact_color as they are)

# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]

#     #convert RGB to HEX code
#     df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
#                           int(i.split(", ")[1]),
#                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

#     df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
#     return df

# def exact_color(card_image, tolerance, zoom):
#     #create dataframe
#     colors_x = extcolors.extract_from_path(card_image, tolerance=tolerance, limit=13)
#     df_color = color_to_df(colors_x)

#     #annotate text
#     list_color = list(df_color['c_code'])
#     list_precent = [int(i) for i in list(df_color['occurence'])]
#     text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color, list_precent)]
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160,120), dpi=10)

#     #donut plot
#     wedges, text = ax1.pie(list_precent,
#                            labels=text_c,
#                            labeldistance=1.05,
#                            colors=list_color,
#                            textprops={'fontsize': 150, 'color':'black'})
#     plt.setp(wedges, width=0.3)

#     #add image in the center of donut plot
#     img = mpimg.imread(card_image)
#     imagebox = OffsetImage(img, zoom=zoom)
#     ab = AnnotationBbox(imagebox, (0, 0))
#     ax1.add_artist(ab)

#     #color palette
#     x_posi, y_posi, y_posi2 = 160, -170, -170
#     for c in list_color:
#         if list_color.index(c) <= 5:
#             y_posi += 180
#             rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
#             ax2.add_patch(rect)
#             ax2.text(x=x_posi+400, y=y_posi+100, s=c, fontdict={'fontsize': 190})
#         else:
#             y_posi2 += 180
#             rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
#             ax2.add_artist(rect)
#             ax2.text(x=x_posi+1400, y=y_posi2+100, s=c, fontdict={'fontsize': 190})

#     fig.set_facecolor('white')
#     ax2.axis('off')
#     plt.imshow(card_image)
#     plt.tight_layout()
#     return plt.show()
# # Function to capture frames from webcam
# def capture_frames():
#     cap = cv2.VideoCapture(0)

#     # Get the dimensions of the webcam frame
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Calculate the position of the rectangle box
#     rect_width = 250
#     rect_height = 350
#     rect_x = int((frame_width - rect_width) / 2)
#     rect_y = int((frame_height - rect_height) / 2)
#     rect_size = (rect_width, rect_height)

#     while True:
#         ret, frame = cap.read()

#         # Display the rectangle box
#         cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)

#         cv2.imshow('Webcam', frame)

#         # Press 'q' to extract colors from the captured region
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             card_region = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]
#             save_card_region(card_region, 'card_image.jpg')
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return 'card_image.jpg'

# # Save the captured card region as an image file
# def save_card_region(card_region, filename):
#     cv2.imwrite(filename, card_region)


# # Call the functions to capture the card region and save it as an image
# image_path = capture_frames()

# # Call the color extraction function with the captured card image
# exact_color(image_path, 12, 2.5)

###########################################################################################################################
# import cv2
# import numpy as np
# import extcolors
# import matplotlib.pyplot as plt
# from PIL import Image
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import matplotlib.image as mpimg
# import matplotlib.patches as patches
# from colormap import rgb2hex
# import pandas as pd

# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]

#     #convert RGB to HEX code
#     df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
#                           int(i.split(", ")[1]),
#                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

#     df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
#     return df

# def exact_color(input_image, resize, tolerance, zoom):
#     #background
#     bg = 'bg.png'
#     fig, ax = plt.subplots(figsize=(192,108),dpi=10)
#     fig.set_facecolor('white')
#     plt.savefig(bg)
#     plt.close(fig)

#     #resize
#     output_width = resize
#     img = Image.open(input_image)
#     if img.size[0] >= resize:
#         wpercent = (output_width/float(img.size[0]))
#         hsize = int((float(img.size[1])*float(wpercent)))
#         img = img.resize((output_width,hsize), Image.ANTIALIAS)
#         resize_name = 'resize_'+ input_image
#         img.save(resize_name)
#     else:
#         resize_name = input_image

#     #create dataframe
#     img_url = resize_name
#     colors_x = extcolors.extract_from_path(img_url, tolerance=tolerance, limit=13)
#     df_color = color_to_df(colors_x)

#     #annotate text
#     list_color = list(df_color['c_code'])
#     list_precent = [int(i) for i in list(df_color['occurence'])]
#     text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color, list_precent)]

#     # Remove green color (used for the rectangle box) from the color palette
#     if '#00ff00' in list_color:
#         index = list_color.index('#00ff00')
#         list_color.pop(index)
#         list_precent.pop(index)
#         text_c.pop(index)

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160,120), dpi=10)

#     #donut plot
#     wedges, text = ax1.pie(list_precent,
#                            labels=text_c,
#                            labeldistance=1.05,
#                            colors=list_color,
#                            textprops={'fontsize': 150, 'color':'black'})
#     plt.setp(wedges, width=0.3)

#     #add image in the center of donut plot
#     img = mpimg.imread(resize_name)
#     imagebox = OffsetImage(img, zoom=zoom)
#     ab = AnnotationBbox(imagebox, (0, 0))
#     ax1.add_artist(ab)

#     #color palette
#     x_posi, y_posi, y_posi2 = 160, -170, -170
#     for c in list_color:
#         if list_color.index(c) <= 5:
#             y_posi += 180
#             rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
#             ax2.add_patch(rect)
#             ax2.text(x=x_posi+400, y=y_posi+100, s=c, fontdict={'fontsize': 190})
#         else:
#             y_posi2 += 180
#             rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
#             ax2.add_artist(rect)
#             ax2.text(x=x_posi+1400, y=y_posi2+100, s=c, fontdict={'fontsize': 190})

#     fig.set_facecolor('white')
#     ax2.axis('off')
#     bg = plt.imread('bg.png')
#     plt.imshow(bg)
#     plt.tight_layout()
#     return plt.show()


# # Function to capture card region from webcam
# def capture_card_region():
#     cap = cv2.VideoCapture(0)

#     # Set the dimensions of the card region
#     card_width = 400
#     card_height = 300

#     while True:
#         ret, frame = cap.read()

#         # Display the frame
#         cv2.imshow('Webcam', frame)

#         # Calculate the center coordinates of the frame
#         frame_width = frame.shape[1]
#         frame_height = frame.shape[0]
#         center_x = frame_width // 2
#         center_y = frame_height // 2

#         # Calculate the top-left corner coordinates of the card region
#         x = center_x - (card_width // 2)
#         y = center_y - (card_height // 2)

#         # Draw a rectangle around the card region (without including it in color extraction)
#         cv2.rectangle(frame, (x, y), (x + card_width, y + card_height), (0, 255, 0), 2)
#         # Draw a filled rectangle outside the card region (to hide the background)
#         cv2.rectangle(frame, (0, 0), (frame_width, y), (0, 0, 0), -1)
#         cv2.rectangle(frame, (0, y + card_height), (frame_width, frame_height), (0, 0, 0), -1)

#         # Extract the card region
#         card_region = frame[y:y + card_height, x:x + card_width]

#         # Press 'q' to exit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

#     return card_region


# # Save the captured card region as an image file
# def save_card_region(card_region, filename):
#     cv2.imwrite(filename, card_region)


# # Call the functions to capture the card region and save it as an image
# card_region = capture_card_region()
# save_card_region(card_region, 'card_image.jpg')

# # Call the color extraction function with the captured card image
# exact_color('card_image.jpg', 900, 12, 2.5)


#### 

# import cv2
# import numpy as np
# import extcolors
# import matplotlib.pyplot as plt
# from PIL import Image
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import matplotlib.image as mpimg
# import matplotlib.patches as patches
# from colormap import rgb2hex
# import pandas as pd

# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]

#     #convert RGB to HEX code
#     df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
#                           int(i.split(", ")[1]),
#                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

#     df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
#     return df

# def exact_color(input_image, resize, tolerance, zoom):
#     #background
#     bg = 'bg.png'
#     fig, ax = plt.subplots(figsize=(192,108),dpi=10)
#     fig.set_facecolor('white')
#     plt.savefig(bg)
#     plt.close(fig)

#     #resize
#     output_width = resize
#     img = Image.open(input_image)
#     if img.size[0] >= resize:
#         wpercent = (output_width/float(img.size[0]))
#         hsize = int((float(img.size[1])*float(wpercent)))
#         img = img.resize((output_width,hsize), Image.ANTIALIAS)
#         resize_name = 'resize_'+ input_image
#         img.save(resize_name)
#     else:
#         resize_name = input_image

#     #create dataframe
#     img_url = resize_name
#     colors_x = extcolors.extract_from_path(img_url, tolerance=tolerance, limit=13)
#     df_color = color_to_df(colors_x)

#     #annotate text
#     list_color = list(df_color['c_code'])
#     list_precent = [int(i) for i in list(df_color['occurence'])]
#     text_c = [c + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color, list_precent)]
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160,120), dpi=10)

#     #donut plot
#     wedges, text = ax1.pie(list_precent,
#                            labels=text_c,
#                            labeldistance=1.05,
#                            colors=list_color,
#                            textprops={'fontsize': 150, 'color':'black'})
#     plt.setp(wedges, width=0.3)

#     #add image in the center of donut plot
#     img = mpimg.imread(resize_name)
#     imagebox = OffsetImage(img, zoom=zoom)
#     ab = AnnotationBbox(imagebox, (0, 0))
#     ax1.add_artist(ab)

#     #color palette
#     x_posi, y_posi, y_posi2 = 160, -170, -170
#     for c in list_color:
#         if list_color.index(c) <= 5:
#             y_posi += 180
#             rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
#             ax2.add_patch(rect)
#             ax2.text(x=x_posi+400, y=y_posi+100, s=c, fontdict={'fontsize': 190})
#         else:
#             y_posi2 += 180
#             rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
#             ax2.add_artist(rect)
#             ax2.text(x=x_posi+1400, y=y_posi2+100, s=c, fontdict={'fontsize': 190})

#     fig.set_facecolor('white')
#     ax2.axis('off')
#     bg = plt.imread('bg.png')
#     plt.imshow(bg)
#     plt.tight_layout()
#     return plt.show()


# # Function to capture card region from webcam
# def capture_card_region():
#     cap = cv2.VideoCapture(0)

#     # Set the dimensions of the card region
#     card_width = 400
#     card_height = 300

#     while True:
#         ret, frame = cap.read()

#         # Display the frame
#         cv2.imshow('Webcam', frame)

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply thresholding to segment the card region
#         _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#         # Find contours in the thresholded image
#         contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Find the largest contour (presumably the card region)
#         if contours:
#             card_contour = max(contours, key=cv2.contourArea)

#             # Get the bounding rectangle of the card contour
#             x, y, w, h = cv2.boundingRect(card_contour)

#             # Check if the bounding rectangle matches the card size
#             if w >= card_width and h >= card_height:
#                 # Extract the card region
#                 card_region = frame[y:y + h, x:x + w]

#                 # Release the webcam and close the windows
#                 cap.release()
#                 cv2.destroyAllWindows()

#                 return card_region

#         # Press 'q' to exit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# # Save the captured card region as an image file
# def save_card_region(card_region, filename):
#     cv2.imwrite(filename, card_region)


# # Call the functions to capture the card region and save it as an image
# card_region = capture_card_region()
# save_card_region(card_region, 'card_image.jpg')

# # Call the color extraction function with the captured card image
# exact_color('card_image.jpg', 900, 12, 2.5)











##################################################################################################################################################
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import matplotlib.image as mpimg
# from PIL import Image
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import cv2
# import extcolors
# from colormap import rgb2hex
# import webcolors

# def color_to_df(input):
#     colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
#     df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
#     df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]
    
#     df_color_up = []
#     df_color_name = []
#     for i in df_rgb:
#         r, g, b = [int(x) for x in i[1:-1].split(", ")]
#         hex_code = rgb2hex(r, g, b)
#         color_name = get_color_name(hex_code)
#         df_color_up.append(hex_code)
#         df_color_name.append(color_name)
    
#     df = pd.DataFrame(zip(df_color_up, df_color_name, df_percent), columns=['c_code','c_name','occurence'])
#     return df

# def get_color_name(hex_code):
#     try:
#         closest_name = webcolors.rgb_to_name(webcolors.hex_to_rgb(hex_code))
#         return closest_name
#     except ValueError:
#         return "Unknown"

# def exact_color(input_image, resize, tolerance, zoom):
#     bg = 'bg.png'
#     fig, ax = plt.subplots(figsize=(192,108),dpi=10)
#     fig.set_facecolor('white')
#     plt.savefig(bg)
#     plt.close(fig)
    
#     output_width = resize
#     img = Image.open(input_image)
#     if img.size[0] >= resize:
#         wpercent = (output_width/float(img.size[0]))
#         hsize = int((float(img.size[1])*float(wpercent)))
#         img = img.resize((output_width,hsize), Image.ANTIALIAS)
#         resize_name = 'resize_'+ input_image
#         img.save(resize_name)
#     else:
#         resize_name = input_image
    
#     img_url = resize_name
#     colors_x = extcolors.extract_from_path(img_url, tolerance=tolerance, limit=13)
#     df_color = color_to_df(colors_x)
    
#     list_color = list(df_color['c_code'])
#     list_name = list(df_color['c_name'])
#     list_precent = [int(i) for i in list(df_color['occurence'])]
#     text_c = [c + ' (' + n + ') ' + str(round(p*100/sum(list_precent),1)) +'%' for c, n, p in zip(list_color, list_name, list_precent)]
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160,120), dpi=10)
    
#     wedges, text = ax1.pie(list_precent,
#                            labels=text_c,
#                            labeldistance=1.05,
#                            colors=list_color,
#                            textprops={'fontsize': 150, 'color':'black'})
#     plt.setp(wedges, width=0.3)

#     img = mpimg.imread(resize_name)
#     imagebox = OffsetImage(img, zoom=zoom)
#     ab = AnnotationBbox(imagebox, (0, 0))
#     ax1.add_artist(ab)
    
#     x_posi, y_posi, y_posi2 = 160, -170, -170
#     for c, n in zip(list_color, list_name):
#         if list_color.index(c) <= 5:
#             y_posi += 180
#             rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
#             ax2.add_patch(rect)
#             ax2.text(x=x_posi+400, y=y_posi+100, s=n, fontdict={'fontsize': 100})
#         else:
#             y_posi2 += 180
#             rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
#             ax2.add_artist(rect)
#             ax2.text(x=x_posi+1400, y=y_posi2+100, s=n, fontdict={'fontsize': 100})

#     fig.set_facecolor('white')
#     ax2.axis('off')
#     bg = plt.imread('bg.png')
#     plt.imshow(bg)       
#     plt.tight_layout()
#     plt.show()


# exact_color('dp_pancard.jpg', 900, 24, 2.5)
