import xml.etree.ElementTree as ET
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os

class TripInfoExtractor:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.waiting_times = []

    def merge_xml_files(self, output_file):
        # 创建根元素
        root = ET.Element("tripinfos")
        
        # 遍历文件夹中的所有文件
        for file_name in os.listdir(self.folder_path):
            if file_name.startswith("tripinfos") and file_name.endswith(".xml"):
                file_path = os.path.join(self.folder_path, file_name)
                
                # 解析XML文件并合并
                tree = ET.parse(file_path)
                file_root = tree.getroot()
                
                for tripinfo in file_root.findall('tripinfo'):
                    root.append(tripinfo)
        
        # 保存合并后的XML文件
        tree = ET.ElementTree(root)
        tree.write(output_file)
        print(f"All files merged into {output_file}")

    def extract_waiting_times(self, merged_file):
        # 解析合并后的XML文件
        tree = ET.parse(merged_file)
        root = tree.getroot()

        # 提取所有的waitingTime值
        for tripinfo in root.findall('tripinfo'):
            waiting_time = tripinfo.get('waitingTime')
            if waiting_time is not None:
                self.waiting_times.append(float(waiting_time))

    def save_to_csv(self, output_csv_path):
        # 将waitingTime数据保存到CSV文件
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['waitingTime'])  # 写入表头
            for waiting_time in self.waiting_times:
                writer.writerow([waiting_time])
        print(f"Data successfully saved to {output_csv_path}")

    def display_data(self):
        # 使用pandas DataFrame显示数据
        df = pd.DataFrame(self.waiting_times, columns=['waitingTime'])
        return df
    
    def plot_histogram(self, output_image_path):
        # 设置区间
        bins = list(range(0, 101, 10)) + [float('inf')]
        labels = [f'{i}-{i+10}' for i in range(0, 100, 10)] + ['>100']

        # 创建新列，将等待时间分配到区间
        df = pd.DataFrame(self.waiting_times, columns=['waitingTime'])
        df['binned'] = pd.cut(df['waitingTime'], bins=bins, labels=labels, right=False)

        # 绘制频数分布直方图
        plt.figure(figsize=(10, 6))
        df['binned'].value_counts(sort=False).plot(kind='bar', edgecolor='black')
        plt.title('Distribution of Waiting Times')
        plt.xlabel('Waiting Time Interval')
        plt.ylabel('Frequency')
        plt.grid(axis='y')
        plt.savefig(output_image_path)
        plt.show()
        print(f"Histogram saved to {output_image_path}")

    def plot_pie_chart(self, output_image_path):
        # 设置区间
        bins = [0, 100, float('inf')]
        labels = ['<=100', '>100']

        # 创建新列，将等待时间分配到区间
        df = pd.DataFrame(self.waiting_times, columns=['waitingTime'])
        df['binned'] = pd.cut(df['waitingTime'], bins=bins, labels=labels, right=False)

        # 绘制饼图
        plt.figure(figsize=(10, 6))
        df['binned'].value_counts(sort=False).plot(kind='pie', autopct='%1.1f%%')
        plt.title('Distribution of Waiting Times')
        plt.ylabel('')  # 隐藏ylabel
        plt.savefig(output_image_path)
        plt.show()
        print(f"Pie chart saved to {output_image_path}")


if __name__ == "__main__":
   

   # 示例使用
    folder_path = r'D:\trg1vr\sumo-rl-main\sumo-rl-main\test-trip'  # 文件夹路径
    merged_file = 'merged_tripinfos.xml'
    output_csv_path = 'waiting_times.csv'
    output_image_path = 'waiting_times_histogram.png'
    output_pie_chart_path = 'waiting_times_pie_chart.png'

    extractor = TripInfoExtractor(folder_path)
    extractor.merge_xml_files(merged_file)
    extractor.extract_waiting_times(merged_file)
    extractor.save_to_csv(output_csv_path)
    df = extractor.display_data()
    extractor.plot_histogram(output_image_path)
    extractor.plot_pie_chart(output_pie_chart_path)