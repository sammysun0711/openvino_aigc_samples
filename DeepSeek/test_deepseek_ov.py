import openvino_genai
import queue
import threading
import argparse

# optimum-cli export openvino --model DeepSeek-R1-Distill-Qwen-1.5B --weight-format fp16 DeepSeek-R1-Distill-Qwen-1.5B-OV-FP16 --task text-generation-with-past --trust-remote-code
# optimum-cli export openvino --model DeepSeek-R1-Distill-Qwen-1.5B --weight-format int4 --group-size 64 --ratio 1.0 DeepSeek-R1-Distill-Qwen-1.5B-OV-INT4 --task text-generation-with-past --trust-remote-code
# optimum-cli export openvino --model DeepSeek-R1-Distill-Qwen-7B --weight-format int4 --group-size 64 --ratio 1.0 DeepSeek-R1-Distill-Qwen-7B-OV-INT4 --task text-generation-with-past --trust-remote-code
# optimum-cli export openvino --model DeepSeek-R1-Distill-Llama-8B --weight-format int4 --group-size 64 --ratio 1.0 DeepSeek-R1-Distill-Llama-8B-OV-INT4 --task text-generation-with-past --trust-remote-code

chat_templates = {
    "DeepSeek-R1": {
        "code-gen": "system\n下面这段的代码的效率很低，且没有处理边界情况。请先解释这段代码的问题与解决方法，然后进行优化：\n{prompt}",
        "code-explain": "system\n请解释下面这段代码的逻辑，并说明完成了什么功能：\n{prompt}",
        "content-classification": "system\n#### 定位\n- 智能助手名称：新闻分类专家\n- 主要任务：对输入的新闻文本进行自动分类，识别其所属的新闻种类。\n\n#### 能力\n- 文本分析 ：能够准确分析新闻文本的内容和结构。\n- 分类识: 根据分析结果，将新闻文本分类到预定义的种类中。\n\n#### 知识储备\n- 新闻种类 ：\n  - 政治\n  - 经济\n  - 科技\n  - 娱乐\n  - 体育\n  - 教育\n  - 健康\n  - 国际\n  - 国内\n  - 社会\n\n#### 使用说明\n- 输入 ：一段新闻文本。\n- 输出 ：只输出新闻文本所属的种类，不需要额外解释。user\n: {prompt}",
        "structure-summary": 'system\n用户将提供给你一段新闻内容，请你分析新闻内容，并提取其中的关键信息，以 JSON 的形式输出，输出的 JSON 需遵守以下的格式：\n\n\n "entiry": <新闻实体>,\n "time": <新闻时间，格式为 YYYY-mm-dd HH:MM:SS，没有请填 null>,\n "summary": <新闻内容总结>\n\': {prompt}',
        "outline-gen": "system\n你是一位文本大纲生成专家，擅长根据用户的需求创建一个有条理且易于扩展成完整文章的大纲，你拥有强大的主题分析能力，能准确提取关键信息和核心要点。具备丰富的文案写作知识储备，熟悉各种文体和题材的文案大纲构建方法。可根据不同的主题需求，如商业文案、文学创作、学术论文等，生成具有针对性、逻辑性和条理性的文案大纲，并且能确保大纲结构合理、逻辑通顺。该大纲应该包含以下部分：\n引言：介绍主题背景，阐述撰写目的，并吸引读者兴趣。\n主体部分：第一段落：详细说明第一个关键点或论据，支持观点并引用相关数据或案例。\n第二段落：深入探讨第二个重点，继续论证或展开叙述，保持内容的连贯性和深度。\n第三段落：如果有必要，进一步讨论其他重要方面，或者提供不同的视角和证据。\n结论：总结所有要点，重申主要观点，并给出有力的结尾陈述，可以是呼吁行动、提出展望或其他形式的收尾。\n创意性标题：为文章构思一个引人注目的标题，确保它既反映了文章的核心内容又能激发读者的好奇心。user\n: {prompt}",
        "slogan-gen": "system\n你是一个宣传标语专家，请根据用户需求设计一个独具创意且引人注目的宣传标语，需结合该产品/活动的核心价值和特点，同时融入新颖的表达方式或视角。请确保标语能够激发潜在客户的兴趣，并能留下深刻印象，可以考虑采用比喻、双关或其他修辞手法来增强语言的表现力。标语应简洁明了，需要朗朗上口，易于理解和记忆，一定要押韵，不要太过书面化。只输出宣传标语，不用解释。user\n: {prompt}",
        "translation": "system\n你是一个中英文翻译专家，将用户输入的中文翻译成英文，或将用户输入的英文翻译成中文。对于非中文内容，它将提供中文翻译结果。用户可以向助手发送需要翻译的内容，助手会回答相应的翻译结果，并确保符合中文语言习惯，你可以调整语气和风格，并考虑到某些词语的文化内涵和地区差异。同时作为翻译家，需将原文翻译成具有信达雅标准的译文。'信' 即忠实于原文的内容与意图；'达' 意味着译文应通顺易懂，表达清晰；'雅' 则追求译文的文化审美和语言的优美。目标是创作出既忠于原作精神，又符合目标语言文化和读者审美的翻译。user\n: {prompt}",
    }
}


class IterableStreamer(openvino_genai.StreamerBase):
    """
    A custom streamer class for handling token streaming and detokenization with buffering.

    Attributes:
        tokenizer (Tokenizer): The tokenizer used for encoding and decoding tokens.
        tokens_cache (list): A buffer to accumulate tokens for detokenization.
        text_queue (Queue): A synchronized queue for storing decoded text chunks.
        print_len (int): The length of the printed text to manage incremental decoding.
    """

    def __init__(self, tokenizer):
        """
        Initializes the IterableStreamer with the given tokenizer.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for encoding and decoding tokens.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_cache = []
        self.text_queue = queue.Queue()
        self.print_len = 0
        self.decoded_lengths = []

    def __iter__(self):
        """
        Returns the iterator object itself.
        """
        return self

    def __next__(self):
        """
        Returns the next value from the text queue.

        Returns:
            str: The next decoded text chunk.

        Raises:
            StopIteration: If there are no more elements in the queue.
        """
        value = (
            self.text_queue.get()
        )  # get() will be blocked until a token is available.
        if value is None:
            raise StopIteration
        return value

    def get_stop_flag(self):
        """
        Checks whether the generation process should be stopped.

        Returns:
            bool: Always returns False in this implementation.
        """
        return False

    def put_word(self, word: str):
        """
        Puts a word into the text queue.

        Args:
            word (str): The word to put into the queue.
        """
        self.text_queue.put(word)

    def put(self, token_id: int) -> bool:
        """
        Processes a token and manages the decoding buffer. Adds decoded text to the queue.

        Args:
            token_id (int): The token_id to process.

        Returns:
            bool: True if generation should be stopped, False otherwise.
        """
        self.tokens_cache.append(token_id)
        text = self.tokenizer.decode(self.tokens_cache)
        self.decoded_lengths.append(len(text))

        word = ""
        delay_n_tokens = 3
        if len(text) > self.print_len and "\n" == text[-1]:
            # Flush the cache after the new line symbol.
            word = text[self.print_len :]
            self.tokens_cache = []
            self.decoded_lengths = []
            self.print_len = 0
        elif len(text) > 0 and text[-1] == chr(65533):
            # Don't print incomplete text.
            self.decoded_lengths[-1] = -1
        elif len(self.tokens_cache) >= delay_n_tokens:
            print_until = self.decoded_lengths[-delay_n_tokens]
            if print_until != -1 and print_until > self.print_len:
                # It is possible to have a shorter text after adding new token.
                # Print to output only if text length is increased and text is complete (print_until != -1).
                word = text[self.print_len : print_until]
                self.print_len = print_until
        self.put_word(word)

        if self.get_stop_flag():
            # When generation is stopped from streamer then end is not called, need to call it here manually.
            self.end()
            return True  # True means stop generation
        else:
            return False  # False means continue generation

    def end(self):
        """
        Flushes residual tokens from the buffer and puts a None value in the queue to signal the end.
        """
        text = self.tokenizer.decode(self.tokens_cache)
        if len(text) > self.print_len:
            word = text[self.print_len :]
            self.put_word(word)
            self.tokens_cache = []
            self.print_len = 0
        self.put_word(None)


class ChunkStreamer(IterableStreamer):

    def __init__(self, tokenizer, tokens_len):
        super().__init__(tokenizer)
        self.tokens_len = tokens_len

    def put(self, token_id: int) -> bool:
        if (len(self.tokens_cache) + 1) % self.tokens_len != 0:
            self.tokens_cache.append(token_id)
            self.decoded_lengths.append(-1)
            return False
        return super().put(token_id)


def stream_chat(pipe, prompt, max_tokens, model_type="qwen"):
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = max_tokens
    if model_type == "qwen":
        config.eos_token_id = 151643
        config.stop_token_ids = {151643, 151647}
    elif model_type == "llama":
        config.eos_token_id = 128001
        config.stop_token_ids = {128001}

    config.do_sample = False
    config.repetition_penalty = 1.1
    config.top_k = 50
    config.top_p = 0.95
    text_print_streamer = IterableStreamer(pipe.get_tokenizer())
    printer_thread = threading.Thread(
        target=pipe.generate, args=(prompt, config, text_print_streamer)
    )
    printer_thread.start()

    return text_print_streamer


def main():
    parser = argparse.ArgumentParser(
        "DeepSeek-R1 Distll Model inference with OpenVNIO",
        add_help=True,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        type=str,
        default="DeepSeek-R1-Distill-Qwen-1.5B-OV-INT4",
        help="Model folder including DeepSeek-R1 Distill OpenVINO Models",
    )
    parser.add_argument(
        "-d", "--device", default="CPU", type=str, help="Inference device"
    )
    parser.add_argument(
        "-cd",
        "--cache_dir",
        default="model_cache",
        type=str,
        help="Folder to save model cache",
    )
    parser.add_argument(
        "-mnt",
        "--max_new_tokens",
        default=1024,
        type=int,
        help="Specify maximum generated tokens counter",
    )

    args = parser.parse_args()
    device = args.device  # GPU can be used as well
    model_dir = args.model_dir
    cache_dir = args.cache_dir
    max_new_tokens = args.max_new_tokens
    model_type = ""
    if "qwen" in str(model_dir).lower():
        model_type = "qwen"
    elif "llama" in str(model_dir).lower():
        model_type = "llama"
    ov_config = {"CACHE_DIR": cache_dir}
    pipe = openvino_genai.LLMPipeline(model_dir, device, **ov_config)

    prompts = {
        "code-gen": "```\ndef fib(n):\n    if n <= 2:\n        return n\n    return fib(n-1) + fib(n-2)\n```",
        "code-explain": "```\n// weight数组的大小 就是物品个数\nfor(int i = 1; i < weight.size(); i++) { // 遍历物品\n    for(int j = 0; j <= bagweight; j++) { // 遍历背包容量\n        if (j < weight[i]) dp[i][j] = dp[i - 1][j];\n else dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i]);\n    }\n}\n```",
        "content-classification": "美国太空探索技术公司（SpaceX）的猎鹰9号运载火箭（Falcon 9）在经历美国联邦航空管理局（Federal Aviation Administration，FAA）短暂叫停发射后，于当地时间8月31日凌晨重启了发射任务。",
        "structure-summary": "8月31日，一枚猎鹰9号运载火箭于美国东部时间凌晨3时43分从美国佛罗里达州卡纳维拉尔角发射升空，将21颗星链卫星（Starlink）送入轨道。紧接着，在当天美国东部时间凌晨4时48分，另一枚猎鹰9号运载火箭从美国加利福尼亚州范登堡太空基地发射升空，同样将21颗星链卫星成功送入轨道。两次发射间隔65分钟创猎鹰9号运载火箭最短发射间隔纪录。\n\n美国联邦航空管理局于8月30日表示，尽管对太空探索技术公司的调查仍在进行，但已允许其猎鹰9号运载火箭恢复发射。目前，双方并未透露8月28日助推器着陆失败事故的详细信息。尽管发射已恢复，但原计划进行五天太空活动的“北极星黎明”（Polaris Dawn）任务却被推迟。美国太空探索技术公司为该任务正在积极筹备，等待美国联邦航空管理局的最终批准后尽快进行发射。",
        "outline-gen": "请帮我生成'中国农业情况'这篇文章的大纲。",
        "slogan-gen": "请生成'希腊酸奶'的宣传标语。",
        "translation": "牛顿第一定律：任何一个物体总是保持静止状态或者匀速直线运动状态，直到有作用在它上面的外力迫使它改变这种状态为止。 如果作用在物体上的合力为零，则物体保持匀速直线运动。 即物体的速度保持不变且加速度为零。",
    }

    template = chat_templates.get("DeepSeek-R1")
    for task, prompt in prompts.items():
        print(
            f"============================== Run task: {task} =============================="
        )
        print("prompt:\n", prompt)
        question = template.get(task).format_map({"prompt": prompt})
        print("response:\n")
        for output in stream_chat(pipe, question, max_new_tokens, model_type):
            print(output, end="", flush=True)
        print("\n")
        pipe.finish_chat()


if "__main__" == __name__:
    main()
