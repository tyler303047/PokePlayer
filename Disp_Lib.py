import time

import Adafruit_Nokia_LCD as LCD
#from Adafruit_Nokia_LCD import *
import Adafruit_GPIO.SPI as SPI
#from Adafruit_GPIO.GPIO import *

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

##def LCD_init():
##    DC = 23
##    RST = 24
##    SPI_PORT = 0
##    SPI_DEVICE = 0
##    Chary = False
##
##    disp = LCD.PCD8544(DC, RST, spi=SPI.SpiDev(SPI_PORT,SPI_DEVICE,max_speed_hz=4000000))
##    disp.begin(contrast=60)
##
##    disp.clear()
##    disp.display()
##
##def LCD_Draw(P, B, S):
##    
##    image = Image.new('1', (LCD.LCDWIDTH, LCD.LCDHEIGHT))
##
##    draw = ImageDraw.Draw(image)
##
##    # Draw a white filled box to clear the image.
##    draw.rectangle((0,0,LCD.LCDWIDTH,LCD.LCDHEIGHT), outline=255, fill=255)
##
##    font = ImageFont.load_default()
##
##    draw.text((0,2), 'Find: ' + 'Patrat' if P else 'Lillipup' , font=font)
##    draw.text((0,15), 'Cap: ' + 'yes' if not B else 'no' , font=font) 
##    draw.text((0,30), S , font=font)
##
##
##    disp.image(image)
##    disp.display()


def LCD_Draw(P, B, S):
    DC = 23
    RST = 24
    SPI_PORT = 0
    SPI_DEVICE = 0

    disp = LCD.PCD8544(DC, RST, spi=SPI.SpiDev(SPI_PORT,SPI_DEVICE,max_speed_hz=4000000))
    disp.begin(contrast=60)

    disp.clear()
    disp.display()
    image = Image.new('1', (LCD.LCDWIDTH, LCD.LCDHEIGHT))

    draw = ImageDraw.Draw(image)

    # Draw a white filled box to clear the image.
    draw.rectangle((0,0,LCD.LCDWIDTH,LCD.LCDHEIGHT), outline=255, fill=255)

    font = ImageFont.load_default()

    draw.text((0,2), 'Find: ' + 'Patrat' if P else 'Lillipup' , font=font)
    draw.text((0,15), 'Cap: ' + 'yes' if not B else 'no' , font=font) 
    draw.text((0,30), S , font=font)


    disp.image(image)
    disp.display()
