import openvino_genai
import queue
import threading
import argparse
import time
# optimum-cli export openvino --model DeepSeek-R1-Distill-Qwen-1.5B --weight-format fp16 DeepSeek-R1-Distill-Qwen-1.5B-OV-FP16 --task text-generation-with-past --trust-remote-code
# optimum-cli export openvino --model DeepSeek-R1-Distill-Qwen-1.5B --weight-format int4 --group-size 64 --ratio 1.0 DeepSeek-R1-Distill-Qwen-1.5B-OV-INT4 --task text-generation-with-past --trust-remote-code
# optimum-cli export openvino --model DeepSeek-R1-Distill-Qwen-7B --weight-format int4 --group-size 64 --ratio 1.0 DeepSeek-R1-Distill-Qwen-7B-OV-INT4 --task text-generation-with-past --trust-remote-code
# optimum-cli export openvino --model DeepSeek-R1-Distill-Llama-8B --weight-format int4 --group-size 64 --ratio 1.0 DeepSeek-R1-Distill-Llama-8B-OV-INT4 --task text-generation-with-past --trust-remote-code


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
        default=4000,
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
    ov_config = {"CACHE_DIR": cache_dir,
                 "DYNAMIC_QUANTIZATION_GROUP_SIZE": "0",
                 "KV_CACHE_PRECISION": "f16"}

    pipe = openvino_genai.LLMPipeline(model_dir, device, **ov_config)

    prompt_list = [
    #"张三参加公司年会抽奖，抽奖规则是系统从数字1，2，3，4，5，6，7，8中选择4个不同的数字来组成目标组，张三从这些数字里也选择4个不重复的数字。如果他选择的数字中至少有两个与目标组中的两个相同，他将获得一个奖项；如果他选择的四个数字全部与目标组的数字相同，他将获得年会的特等奖。已知他获得了一个奖项，而他获得特等奖的概率是 m/n，其中m和n是互质的正整数。求m+n。",
    #"一个年级有90个小学生，参加运动会，19个人参加了短跑，36个人参加了跳高，55个人参加了拔河，90个人都参加了踢毽子。有43个小学生正好参加了四项活动中的两项，有23个小学生正好参加了四项活动中的三项。问参加了全部四项活动的小学生有几个人？",
    "每个周末，小王都会步行9公里健身，并在之后进行固定时间的拉伸。当她以每小时s公里的恒定速度行走时，步行需要4小时，包括拉伸用的的t分钟。当她以每小时s + 2公里的速度行走时，步行需要2小时24分钟，也包括拉伸用的t分钟。假设小王以每小时s + 1/2公里的速度行走。求她健身总共需要多少分钟，包括步行时间，以及最后拉伸的t分钟固定时间。"]
    
    start = time.time()
    for i, prompt in enumerate(prompt_list): 
        print(f"Iteration {i} start with prompt: \n{prompt}")
        for output in stream_chat(pipe, prompt, max_new_tokens, model_type):
            print(output, end="", flush=True)
        print("\n")
        pipe.finish_chat()
    duration = time.time() - start
    print(f"Generation elapsed time {duration:.2f}s")


if "__main__" == __name__:
    main()
