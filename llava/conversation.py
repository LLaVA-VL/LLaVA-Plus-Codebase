import dataclasses
from enum import auto, Enum
import os
import re
from typing import List, Tuple
import torchvision.transforms.functional as F
from PIL import Image


def parse_tool_output(text):
    try:
        pattern = r'"thoughts🤔"(.*)"actions🚀"(.*)"value👉"(.*)'
        matches = re.findall(pattern, text, re.DOTALL)
        assert len(matches) == 1, f"len(matches)={len(matches)}"
        assert len(matches[0]) == 3, f"len(matches[0])={len(matches[0])}"
    except Exception as e:
        # print(e)
        matches = None
        return matches
    return matches


def make_it_small_html(text):
    return f'<span style="font-size: 12px; color: gray;line-height: 1.0;">{text}</span>'


def get_hr_html():
    return f'<hr width="100%" size="1" color="silver" align="center">'


def get_placehold(text):
    if text[-1] == "▌":
        text = text[:-1]

    res = '"thinking'
    timenow = len(text) % 21
    num_point = timenow // 3
    for i in range(num_point):
        res += "."
    res += '"'
    return res


def parse_msg(msg):
    if len(msg) == 3:
        return msg[0], msg[1], msg[2], None
    if len(msg) == 4:
        return msg[0], msg[1], msg[2], msg[3]
    raise ValueError(f"Invalid msg with len {len(msg)}: {msg}")


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _, _ = parse_msg(message)
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _, _ = parse_msg(message)
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _, _ = parse_msg(message)
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            def wrap_sys(msg): return f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            def wrap_inst(msg): return f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _, _ = parse_msg(message)
                    if i == 0:
                        message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _, _ = parse_msg(message)
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if len(self.roles) > 2 and role == self.roles[2]:
                continue
            if role == self.roles[0]:
                # if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    # msg, image, image_process_mode = msg
                    msg, image, image_process_mode, sketch_mask = parse_msg(
                        msg)
                    if image_process_mode == "Pad":
                        def expand2square(pil_img, background_color=(122, 116, 104)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(
                                    pil_img.mode, (width, width), background_color)
                                result.paste(
                                    pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(
                                    pil_img.mode, (height, height), background_color)
                                result.paste(
                                    pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image)
                    elif image_process_mode in ["Default", "Crop"]:
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((336, 336))
                    elif image_process_mode == "None":
                        pass
                    else:
                        raise ValueError(
                            f"Invalid image_process_mode: {image_process_mode}")
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(
                        min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if longest_edge != max(image.size):
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((W, H))
                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(
                            buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def get_raw_images(self, return_pil=False, image_process_mode=None):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if len(self.roles) > 2 and role == self.roles[2]:
                continue
            if role == self.roles[0]:
                # if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    msg, img, _, sketch_mask = parse_msg(msg)

                    # resize for large images
                    w, h = img.size
                    if max(h, w) > 800:
                        if h > w:
                            new_h = 800
                            new_w = int(w * 800 / h)
                        else:
                            new_w = 800
                            new_h = int(h * 800 / w)
                        # import ipdb; ipdb.set_trace()
                        img = F.resize(img, (new_h, new_w))

                    if return_pil:
                        images.append(img)
                    else:
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(
                            buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def tools_filter_msg(self, msg):
        return msg

    def merge_output(self, ret, with_debug_parameter=False):
        # print(f'with_debug_parameter: {with_debug_parameter}')
        assert isinstance(
            ret, list), "ret should be a list, but got {}".format(type(ret))

        ret_new = []
        i = 0
        while i < len(ret):
            text: str = ret[i][0]

            # # check if previous is "thinking.."
            # if len(ret_new) > 0 and isinstance(ret_new[-1][0], str) and ret_new[-1][0].strip().replace('.', '') == '"thinking"':
            #     ret_new = ret_new[:-1]

            # for some undisplayed message
            if not isinstance(text, str):
                ret_new.append(ret[i])
                i += 1
                continue

            text = text.strip()
            # for the case with image
            if text.startswith('<img src="data:image/png;base64'):
                if len(ret_new) > 0:
                    ret_new[-1] = [ret_new[-1][0] + '\n' + ret[i][0], None]
                else:
                    ret_new.append(ret[i])
                i += 1
                continue

            if text.startswith('"th'):
                # for "thoughts🤔"
                matches = parse_tool_output(text)
                if matches is not None:
                    thought = matches[0][0]
                    action = matches[0][1]
                    value = matches[0][2]

                    action_json = eval(action)
                    # if len(action_json) > 0:
                    if (len(action_json) > 0):
                        # tool use branch
                        res_value = f'"thoughts🤔" {matches[0][0].strip()}\n' +\
                             f'"actions🚀" {matches[0][1].strip()}\n' \
                           +  f'"value👉" {matches[0][2].strip()}'
                        res_value = make_it_small_html(res_value)

                        # explore next
                        matches_next2 = None
                        if (i + 1 < len(ret)):
                            # get next message
                            text_next: str = ret[i +
                                                 1][0].strip().replace("\n\n", "\n")
                            if len(ret_new) > 0 and "model outputs:" in text_next:
                                # auged ques
                                text_next_html = make_it_small_html(text_next)
                                res_value = res_value + get_hr_html() + text_next_html

                                # explore next2
                                if i + 2 < len(ret):
                                    text_next2: str = ret[i+2][0].strip()
                                    # if text_next2.startswith('"th'):
                                    matches_next2 = parse_tool_output(
                                        text_next2)

                                    if matches_next2 is not None:
                                        text_next2_html = f'"thoughts🤔" {matches_next2[0][0].strip()}\n' + \
                                            f'"actions🚀" {matches_next2[0][1].strip()}\n' + \
                                            f'"value👉"'
                                        text_next2_html = make_it_small_html(
                                            text_next2_html)
                                        res_value = res_value + get_hr_html() + text_next2_html
                                        res_value = res_value + \
                                            f'\n{matches_next2[0][2].strip()}'
                                        i += 1
                                    else:
                                        res_value = res_value + get_hr_html() + make_it_small_html(text_next2)
                                        i += 1
                                i += 1

                        # post process for no debug parameters
                        if not with_debug_parameter:
                            if matches_next2 is not None:
                                res_value = matches_next2[0][2].strip()
                            else:
                                res_value = get_placehold(res_value)

                        # add to ret_new
                        ret_new.append([res_value, None])
                    else:
                        # regular conv branch
                        if with_debug_parameter:
                            res_value = f'"thoughts🤔" {matches[0][0].strip()}\n' +\
                                 f'"actions🚀" {matches[0][1].strip()}\n' \
                               +  f'"value👉"\n'
                            res_value = make_it_small_html(res_value)
                            res_value = res_value + f'{matches[0][2].strip()}'
                        else:
                            res_value = f'{matches[0][2].strip()}'

                        ret_new.append([res_value, None])
                else:
                    if with_debug_parameter:
                        ret_new.append(ret[i])
                    else:
                        ret_new.append([
                            get_placehold(ret[i][0].strip()),
                            None
                        ])
            else:
                ret_new.append(ret[i])
            i += 1

        return ret_new

    def image_to_url(self, image):
        import base64
        from io import BytesIO
        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        max_len, min_len = 800, 400
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
        img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
        return img_str

    def to_gradio_chatbot(self, with_debug_parameter=False):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            # if i % 2 == 0:
            if len(self.roles) > 2 and role == self.roles[2]:
                continue
            # if role == self.roles[0]:
            if 1:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    # msg, image, image_process_mode = msg
                    msg, image, image_process_mode, sketch_mask = parse_msg(
                        msg)
                    if not isinstance(image, list):
                        img_str = self.image_to_url(image)
                        if i == 0:
                            ret.append([img_str, None])
                        msg = msg.replace('<image>', '').strip()
                        if role == self.roles[1]:
                            msg = self.tools_filter_msg(msg)
                        if len(msg) > 0:
                            ret.append([msg, None])
                        if i != 0:
                            ret.append([img_str, None])
                    else:
                        # a list of image
                        if role == self.roles[1]:
                            msg = self.tools_filter_msg(msg)
                            msg = msg.replace('<image>', '').strip()
                        if len(msg) > 0:
                            ret.append([msg, None])
                        for j, img in enumerate(image):
                            img_str = self.image_to_url(img)
                            ret.append([img_str, None])
                else:
                    if role == self.roles[1]:
                        msg = self.tools_filter_msg(msg)
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg

        ret = self.merge_output(ret, with_debug_parameter=with_debug_parameter)
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self, force_str=False):
        def remove_pil(x, force_str):
            if not force_str:
                return x
            
            if isinstance(x, Image.Image):
                return b64_encode(x)
            
            if isinstance(x, list):
                return [remove_pil(y, force_str) for y in x]
            if isinstance(x, tuple):
                return [remove_pil(y, force_str) for y in x]
            if isinstance(x, dict):
                return {k: remove_pil(v, force_str) for k, v in x.items()}
            
            return x
            
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, remove_pil(y[0], force_str=force_str) if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": remove_pil(self.messages, force_str=force_str),
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

# enable this to use a different default conversation by setting the LLAVA_DEFAULT_CONVERSATION environment variable
default_conversation_name = os.getenv(
    "LLAVA_DEFAULT_CONVERSATION", "conv_vicuna_v1")
default_conversation = globals()[default_conversation_name]
print(f"Using conversation: {default_conversation_name}")

conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,

    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,

    "mpt": conv_mpt,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
