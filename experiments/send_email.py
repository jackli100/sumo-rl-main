import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class EmailSender:
    def __init__(self, username, auth_code):
        self.server = 'smtp.qq.com'
        self.port = 465  # SSL端口
        self.username = username
        self.auth_code = auth_code

    def send_email(self, recipient, subject, body):
        # 创建邮件对象
        msg = MIMEMultipart()
        msg['From'] = self.username
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # 连接到 SMTP 服务器
        with smtplib.SMTP_SSL(self.server, self.port) as server:
            server.login(self.username, self.auth_code)
            server.send_message(msg)
            print("Email sent successfully!")

# 使用示例
if __name__ == "__main__":
    # 替换以下信息为你的实际信息
    email_sender = EmailSender('1811743445@qq.com', 'zbfuppehtwtkejfg')
    email_sender.send_email('zl22n23@soton.ac.uk', 'Hello from Python', 'This is a test email sent from Python using QQ mail.')
