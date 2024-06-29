import os
import json
import yaml
import time
import random
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

    
def str_time_prop(start, end, time_format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formatted in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, time_format))
    etime = time.mktime(time.strptime(end, time_format))

    ptime = stime + prop * (etime - stime)

    return time.localtime(ptime)


def add_texts(template, image, font):

    universities_df = pd.read_csv('data/information/universities.csv')
    subjects_df = pd.read_csv('data/information/subjects.csv',
                              encoding="utf-8",
                              dtype={'so_tin_chi': 'str', 'diem': 'str'})
    students_df = pd.read_csv('data/information/students.csv',
                              encoding="utf-8",
                              dtype={'dien_thoai': 'str','lop': 'str'})
    
    with open('data/information/other_info.yaml') as yaml_file:
        other_info = yaml.safe_load(yaml_file)
    
    with open('config/name_of_column.yaml') as yaml_file:
        name_of_column = yaml.safe_load(yaml_file)

    student_info = students_df.iloc[random.randint(0, 4127), :]

    draw = ImageDraw.Draw(image)
    num_filled_answer = random.randint(3, 4)
    for loc in template:
        field = loc['text']

        for key, value in name_of_column.items():
            if field in value:
                field = key

        for i, answer in enumerate(loc['answer_text']):
            if i == num_filled_answer:
                break
            box = answer['box']
            text_width = box[2] - box[0]
            text_height = box[3] - box[1]

            count = 0
            while True:
                if loc['class'] == 'question':
                    if field in universities_df.columns:
                        text = universities_df.loc[random.randint(0, 60), field]
                    elif field in subjects_df.columns:
                        text = subjects_df.loc[random.randint(0, 194), field]
                    elif field in students_df.columns:
                        text = student_info[field]
                    elif field in other_info.keys():
                        text = random.choice(other_info[field])
                    elif field == 'ngay_day_du':
                        text = time.strftime('%d/%m/%Y', str_time_prop("1/1/2010", "31/12/2019", '%d/%m/%Y', random.random()))
                    elif field in ['ngay', 'thang', 'nam']:
                        date = str_time_prop("1/1/2010", "31/12/2019", '%d/%m/%Y', random.random())
                        if field == 'ngay':
                            text = time.strftime('%d', date)
                        elif field == 'thang':
                            text = time.strftime('%m', date)
                        elif field == 'nam':
                            text = time.strftime('%Y', date)
                    elif field == 'so_lon':
                        text = str(random.randint(100, 143))
                    elif field == 'so_nho':
                        text = str(random.randint(0, 12))
                    else:
                        print(field)
                        text = ''
                
                _, _, w, h = draw.textbbox((0, 0), text, font=font)
                if w <= text_width:
                    break
                else:
                    if field in students_df.columns:
                        student_info = students_df.iloc[random.randint(0, 4127), :]
                    
                count += 1
                if count > 10:
                    text = text[:6]
                    _, _, w, h = draw.textbbox((0, 0), text, font=font)
                    break

            draw.text((box[0]+(text_width-w)/2, box[1]+(text_height-h)/2-7), text, fill='black', font=font)

    return image

def main():
    # Test directory
    output_data = 'output'

    if not os.path.exists(f'{output_data}'):
        os.makedirs(f'{output_data}')

    images_dir = 'data/images'
    fonts_dir = 'data/fonts'

    images_files = os.listdir(images_dir)
    fonts_files = os.listdir(fonts_dir)

    for font_name in fonts_files:
        # Load the font
        font = ImageFont.truetype(f"data/fonts/{font_name}", size=50)
        for image_name in images_files:
            
            with open(f"data/templates/{image_name.split(".")[0]}.json", 'r') as file:
                template = json.load(file)
            image = Image.open(f"data/images/{image_name}")

            image = add_texts(template, image, font)

            image.save(f"output/{font_name.split(".")[0]}_{image_name.split(".")[0]}.jpg")

    # test 1
    # font_name = "font_22.ttf"
    # image_name = "don_xin_nhan_diem_i_diem_chua_hoan_tat.jpg"

    # font = ImageFont.truetype(f"data/fonts/{font_name}", size=54)
    # with open(f"data/templates/{image_name.split(".")[0]}.json", 'r') as file:
    #     template = json.load(file)
    # image = Image.open(f"data/images/{image_name}")

    # image = add_texts(template, image, font)

    # image.save(f"output/{font_name.split(".")[0]}_{image_name.split(".")[0]}.jpg")

    # test 2
    # font_name = "font_22.ttf"
    # font = ImageFont.truetype(f"data/fonts/{font_name}", size=50)
    # for image_name in images_files:
    #     print(f'---*---*---*---*---*---*---*---\n{image_name.split(".")[0]}')
    #     with open(f"data/templates/{image_name.split(".")[0]}.json", 'r') as file:
    #         template = json.load(file)
    #     image = Image.open(f"data/images/{image_name}")

    #     image = add_texts(template, image, font)

    #     image.save(f"output/{font_name.split(".")[0]}_{image_name.split(".")[0]}.jpg")


if __name__ == "__main__":
    main()
