from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from browser_use import Agent, Browser, BrowserConfig, Controller, ActionResult
from browser_use.browser.context import BrowserContext
import asyncio
import logging
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Reuse existing browser
browser = Browser(
    config=BrowserConfig(
        # Specify the path to your Chrome executable
        chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',  # macOS path
        # For Windows, typically: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
        # For Linux, typically: '/usr/bin/google-chrome'
    )
)

# Initialize the model
llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0.0,
)
planner_llm = ChatOpenAI(model='o3-mini')
small_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)


# Define sensitive data
# The model will only see the keys (x_name, x_password) but never the actual values
sensitive_data = {
    'x_linkedin_user': 'khoa.nd.thcn@gmail.com', 'x_linkedin_password': '2om-sdE-Dnv-XZT',
    'x_email': 'khoand.pfiev.k49@gmail.com', 'x_password': 'D,m*]5?aZR'}

# Use the placeholder names in your task description
task = '''Hunting best candidate for ai engineer role or llm engineer in linkedin using linkedin account: with x_linkedin_user and x_linkedin_password then access to first 15 potential candidate to extract those candidates infos: profile_url, Name, Role, Location, Experiences, Education; send to x_email on gmail.com, with x_email and x_password'''
# task = '''go to mail.viettel.com.vn login and read first email'''

controller = Controller()

# Content Actions
@controller.registry.action(
    'Extract page content to retrieve specific information from the page, e.g. all company names, a specifc description, all information about, links with companies in structured format or simply links',
)
async def extract_content(goal: str, browser: BrowserContext, page_extraction_llm: BaseChatModel):
    page = await browser.get_current_page()
    import pyhtml2md

    # content = markdownify.markdownify(await page.content())
    content = pyhtml2md.convert(await page.content())

    prompt = 'Your task is to extract the content of the page. You will be given a page and a goal and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format. Extraction goal: {goal}, Page: {page}'
    template = PromptTemplate(input_variables=['goal', 'page'], template=prompt)
    try:
        output = page_extraction_llm.invoke(template.format(goal=goal, page=content))
        msg = f'📄  Extracted from page\n: {output.content}\n'
        logger.info(msg)
        return ActionResult(extracted_content=msg, include_in_memory=True)
    except Exception as e:
        logger.debug(f'Error extracting content: {e}')
        msg = f'📄  Extracted from page\n: {content}\n'
        logger.info(msg)
        return ActionResult(extracted_content=msg)

# Pass the sensitive data to the agent
agent = Agent(
    task=task, 
    llm=llm, 
    use_vision=False, 
    sensitive_data=sensitive_data,
    save_conversation_path="logs/conversation",
    browser=browser,  # Browser instance will be reused
    planner_llm=planner_llm,           # Separate model for planning
    use_vision_for_planner=False,      # Disable vision for planner
    page_extraction_llm=small_llm,
    controller=controller,
    planner_interval=4                 # Plan every 4 steps
)

async def main():
    await agent.run()
    
    # Manually close the browser
    await browser.close()


asyncio.run(main())
