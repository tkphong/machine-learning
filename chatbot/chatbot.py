import aiml
import os
import re

kernel = aiml.Kernel()
bot_brain_file = "chatbot/bot_brain/bot_brain.brn"
startup_file = "chatbot/bot_files/std-startup.xml"

if os.path.isfile(bot_brain_file):
    kernel.bootstrap(brainFile=bot_brain_file)
else:
    kernel.bootstrap(learnFiles=startup_file,
                     commands="load chatbot")
    kernel.saveBrain(bot_brain_file)


def bot_response(message):
    if message == "save":
        kernel.saveBrain(bot_brain_file)
        return "OK:save"
    elif message == "clean":
        os.remove(bot_brain_file)
        return "OK:clean"
    elif message == "reload":
        os.remove(bot_brain_file)
        kernel.bootstrap(learnFiles=startup_file, commands="load chatbot")
        return "OK:reload"
    else:
        message = message.replace("?", "").replace("!", "").replace(".", "")
        message = re.sub("'s", " is", message)
        message = re.sub("o' clock", "o clock", message)
        bot_response = kernel.respond(message)
        return bot_response
