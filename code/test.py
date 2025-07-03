import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from dotenv import load_dotenv

def send_qq_email(subject, content):
    # 配置参数
    load_dotenv()
    mail_host = "smtp.qq.com"       # SMTP服务器
    mail_port = 465                 # SSL端口
    mail_user = os.getenv("SMTP_QQ_EMAIL")   # 你的QQ邮箱
    mail_pass = os.getenv("SMTP_QQ_PASSWORD")   # SMTP授权码（不是邮箱密码！）

    # 创建邮件内容
    msg = MIMEText(content, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = mail_user
    msg['To'] = os.getenv("TO_ADDRESS")

    # 发送邮件
    try:
        smtp = smtplib.SMTP_SSL(mail_host, mail_port)
        smtp.login(mail_user, mail_pass)
        smtp.sendmail(mail_user, [os.getenv("TO_ADDRESS")], msg.as_string())
        print("Email sent successfully")
    except Exception as e:
        print(f"发送失败: {str(e)}")
    finally:
        smtp.quit()


send_qq_email(
    "程序运行完成通知", 
    f"优化已完成" 
    )
