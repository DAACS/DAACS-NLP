from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_MODEL = "gpt-3.5-turbo-1106"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL)

result = llm.invoke("Pretend I am a freshman in college who has just taken a class on how to make better grades. Now please write me a 300 word essay on what I might to to get those better grades.")
print(result)

"""
(.venv) hurricane:daacs-nlp afraser$ python ./src/demo_most_simple.py
content='As a freshman in college, getting better grades is a common goal for many students. After taking a class on how to make better grades, there are several strategies that can be implemented to achieve this goal.\n\nFirst and foremost, it is important to stay organized. Keeping track of assignment due dates, exam dates, and other important deadlines is crucial for success in college. Utilizing a planner or digital calendar can help keep track of these dates and ensure that nothing is missed.\n\nAdditionally, forming good study habits is essential for improving grades. This may include finding a quiet and comfortable study space, breaking up study sessions into manageable chunks, and actively engaging with the material through techniques such as summarizing, questioning, and making connections.\n\nFurthermore, seeking help when needed is important. Whether it be attending office hours, forming study groups, or utilizing tutoring services, there are many resources available to help students succeed. It is important to recognize when additional support is needed and to take advantage of these resources.\n\nAnother important aspect of getting better grades is time management. Balancing academics with other responsibilities can be challenging, but prioritizing tasks and managing time effectively can lead to improved grades. This may involve creating a schedule, setting specific goals, and minimizing distractions.\n\nFinally, taking care of oneself is crucial for academic success. This includes getting enough sleep, eating well, and managing stress. When students prioritize their well-being, they are better able to focus and perform well academically.\n\nIn conclusion, there are several strategies that can be implemented to achieve better grades in college. By staying organized, forming good study habits, seeking help when needed, managing time effectively, and taking care of oneself, students can work towards achieving their academic goals. With dedication and perseverance, better grades are within reach.'
(.venv) hurricane:daacs-nlp afraser$
"""
