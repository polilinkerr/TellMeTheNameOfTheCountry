import os
import gzip
import io
import json
from os.path import dirname, splitext, exists, join, basename, getsize
from http.client import HTTPSConnection
from urllib.parse import quote, unquote, urlparse  #

from reportlab.graphics import renderPDF
from svglib import svglib
from skimage.measure import compare_ssim as ssim
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import ntpath

from pdf2image import convert_from_path
import wikipedia as wiki


urlWikiFlags = "https://en.wikipedia.org/wiki/Gallery_of_sovereign_state_flags"
pathh = os.getcwd()

def flagName_fromURL(path):
    head, tail = ntpath.split(path)
    return tail[:-4]
    #return tail or ntpath.basename(head)


def errorRarpot(fileNamePNG):
    print("ERRRO - can not create - ", fileNamePNG)
    pass


def loadDataBaseFromWiki(url, reload):
    dataBasePath = os.path.join(pathh, "WikiFlags")
    if not exists(dataBasePath):
        os.mkdir(dataBasePath)
    if reload == True:
        wikiPageOfFlags = wiki.page("Gallery_of_sovereign_state_flags")
        listOfFlagsImages = wikiPageOfFlags.images

        flag_dictionary_URL = {}
        flag_dictionary_PNG = {}
        for imageurl in listOfFlagsImages:
            if "Flag_of" in imageurl:
                flag_dictionary_URL[flagName_fromURL(imageurl)] = imageurl  # dodaje do slownika
        # sciaga obrazy

        for country in flag_dictionary_URL.keys():
            fileNameSVG = os.path.join(dataBasePath, country + ".svg")
            if not exists(fileNameSVG):
                print(fileNameSVG)
                flag_svg = fetch_file(flag_dictionary_URL[country])
                with io.open(fileNameSVG, "w", encoding='UTF-8') as f:
                    f.write(flag_svg)

            # if not country in flag_dictionary_SVG:
            #    flag_dictionary_SVG[country] = fileNameSVG

            fileNamePDF = os.path.join(dataBasePath, country + ".pdf")
            if not exists(fileNamePDF):
                drawing = svglib.svg2rlg(fileNameSVG)
                # save as PDF
                base = splitext(fileNameSVG)[0] + '.pdf'
                renderPDF.drawToFile(drawing, base, showBoundary=0)

            fileNamePNG = os.path.join(dataBasePath, country + ".png")
            if not exists(fileNamePNG):
                print("working on  %s - PNG_files" % (country))
                try:
                    images = convert_from_path(fileNamePDF, 500)
                    for i in images:
                        i.save(fileNamePNG, 'PNG')
                except:
                    errorRarpot(fileNamePNG)

            if not country in flag_dictionary_PNG:
                flag_dictionary_PNG[country] = fileNamePNG
    else:
        flag_dictionary_PNG = {}
        listofPNG_Files = [file for file in os.listdir(dataBasePath) if (os.path.isfile(file) & file.endswith("png"))]
        for file in listofPNG_Files:
            fileName = file[:-4]
            flag_dictionary_PNG[fileName] = os.path.join(dataBasePath, fileName)
    return flag_dictionary_PNG


def loadDataBase(url, reload=True):
    pathToJSON = os.path.join(pathh,"flagsDataBase.json")
    if not exists(pathToJSON):
        flag_dictionary_PNG = loadDataBaseFromWiki(url,reload=True)
        with io.open(pathToJSON, "w", encoding='UTF-8') as fh:
            fh.write(json.dumps(flag_dictionary_PNG, ensure_ascii=False))
    else:
        with open(pathToJSON) as json_file:
            flag_dictionary_PNG = json.load(json_file)
    return flag_dictionary_PNG


def flag_url2filename(url):
        path = basename(url)[len("Flag_of_"):]
        path = path.capitalize() # capitalise leading "the_"
        path = unquote(path)

        return path

def fetch_file(url):
        "Get content with some given URL, uncompress if needed."

        parsed = urlparse(url)
        conn = HTTPSConnection(parsed.netloc)
        conn.request("GET", parsed.path)
        r1 = conn.getresponse()
        if (r1.status, r1.reason) == (200, "OK"):
            data = r1.read()
            if r1.getheader("content-encoding") == "gzip":
                zbuf = io.BytesIO(data)
                zfile = gzip.GzipFile(mode="rb", fileobj=zbuf)
                data = zfile.read()
                zfile.close()
            data = data.decode('utf-8')
        else:
            data = None
        conn.close()

        return data

def prepareInput(number):
    listofPNG_Files = [file for file in os.listdir(pathh) if (os.path.isfile(file) & file.endswith("png"))]
    if listofPNG_Files:
        fileName = listofPNG_Files[number]
    else:
        fileName = None

    return os.path.join(pathh,fileName)


def changeTypeOfImage(PNGImage):
    data = PNGImage.astype(np.float64) / np.amax(PNGImage)  # normalize the data to 0 - 1
    data = 255 * data  # Now scale by 255
    img = data.astype(np.uint8)
    return img


def compareFiles(pathToInput, pathToPNG_File):
    #proc = alternativeEngine(pathToInput, pathToPNG_File)
    proc = None
    InputImage = cv2.imread(pathToInput)
    PNGImage = cv2.imread(pathToPNG_File)

    PNGImage = resize(PNGImage, (InputImage.shape[0],InputImage.shape[1]),anti_aliasing=True )
    InputImage_gray = InputImage
    PNGImage_gray = changeTypeOfImage(PNGImage)
    InputImage_gray = cv2.cvtColor(InputImage_gray, cv2.COLOR_BGR2GRAY)
    PNGImage_gray = cv2.cvtColor(PNGImage_gray, cv2.COLOR_BGR2GRAY)

    m = mse(InputImage_gray, PNGImage_gray)
    s = ssim(InputImage, PNGImage, multichannel=True)
    return m, s,proc

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def chooseRightCountry(pathToInput, flag_dictionary_PNG):
    dictOfScores_m = {}
    dictOfScores_s = {}
    for key in flag_dictionary_PNG:
        try:
            m,s,proc = compareFiles(pathToInput,flag_dictionary_PNG[key])
            print(key, " - ", m,s,proc)
            dictOfScores_m[key] = m
            dictOfScores_s[key] = s
        except AttributeError:
            print(key, " - ", None, None, None)
    finalResult = getMinItemFromDict(dictOfScores_m)
    return finalResult[0]

def getMaxItemFromDict(dict):
    max_value = max(dict.values())
    max_keys = [k for k, v in dict.items() if v == max_value]
    return (max_keys,max_value)

def getMinItemFromDict(dict):
    min_value = min(dict.values())
    min_keys = [k for k, v in dict.items() if v == min_value]
    return (min_keys,min_value)


def displayImage(pathToFIle):
    img = mpimg.imread(pathToFIle,0)
    plt.imshow(img)
    plt.show()

def takesmalldicionary(flag_dictionary_PNG,number):
    temp_dict = {}
    listOfKeys = flag_dictionary_PNG.keys()
    iteration = 0
    for i in flag_dictionary_PNG.items():
        if iteration<number:
            temp_dict[i[0]] = i[1]
            iteration=iteration+1
        else:
            break
    return temp_dict

def main():
    flag_dictionary_PNG = loadDataBase(urlWikiFlags,reload=True)
    ######flag_dictionary_PNG_SMALL = takesmalldicionary(flag_dictionary_PNG, 15)
    #####flag_dictionary_PNG = flag_dictionary_PNG_SMALL
    pathToInput = prepareInput(0)
    finalResult = chooseRightCountry(pathToInput,flag_dictionary_PNG)
    print("This is a",finalResult[0])

if __name__ == '__main__':
    main()