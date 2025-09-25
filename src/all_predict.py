# Standard library
import base64
from collections import Counter
from io import BytesIO
import random
import re
import time

# Third-party
import PIL
from google.genai import types
from google.genai.errors import ServerError
from google.genai.types import GenerateContentConfig
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception, RetryCallState

# Local modules
from qwen_vl_utils import process_vision_info
from src.utils import extract_last_class

DIAGRAM_PATH = "frcartoon.png"

def PIL_to_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return image_bytes

GEMINI_SLEEP=10

@retry(
    stop=stop_after_attempt(30),
    wait=wait_exponential(min=2, max=10),
    retry=retry_if_exception_type(ServerError),
    reraise=True,
)
def retry_generate_content(model_to_run, *args, **kwargs):
    return model_to_run.generate_content(*args, **kwargs)

def _is_retryable_openai_error(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        return status in (429, 500, 502, 503, 504)
    return False


def _wait_retry_after_or_expo(retry_state: RetryCallState) -> float:
    exc = retry_state.outcome.exception()
    try:
        resp = getattr(exc, "response", None)
        if resp is not None:
            ra = resp.headers.get("Retry-After")
            if ra:
                return float(ra)
    except Exception:
        pass

    m = re.search(r"in\s+([0-9.]+)s", str(exc))
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    n = max(1, retry_state.attempt_number)
    return min(30.0, (1.5 ** (n - 1))) + random.uniform(0, 0.5)

@retry(
    reraise=True,
    stop=stop_after_attempt(12),
    wait=_wait_retry_after_or_expo,
    retry=retry_if_exception(_is_retryable_openai_error),
)
def _chat_create_with_retry(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def chat_with_retries(*, client, model, messages, max_tokens=300, temperature=0.0, **kwargs):
    return _chat_create_with_retry(
        client=client,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )

def PIL_to_base64(img, format="PNG"):
    buffered = BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def zero_shot_predict_openai(
        client,
        model_name,
        test_dataset,
        query_idx,
        device="",
        system_message="",
        query_text="",
        temperature=0.0,
        reps=1,
        max_new_tokens=512,
        print_text=False,
        image_first=False,
        all_labels=["FR-I","FR-II",],
    ):
    assert system_message != "" and query_text != ""
    
    imgq = test_dataset[query_idx]["image"]
    base64_image = PIL_to_base64(imgq)

    if image_first:
        pass
        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "auto"
                        }
                    },
                    {"type": "text", "text": query_text},
                ]
            }
        ]

    else:

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "auto"
                        }
                    }
                ]
            }
        ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        n=reps,
        max_tokens=max_new_tokens
    )

    answers = Counter()
    raw_answers = []

    for i, choice in enumerate(response.choices, start=1):
        raw = choice.message.content.strip()
        raw_answers.append(raw)
        if print_text:
            print(f"Candidate #{i} raw output:\n{raw}\n{'—'*20}")

        cls = extract_last_class(raw, all_labels)
        answers[cls] += 1

    try:
        final_answer = answers.most_common(1)[0][0]
    except IndexError:
        final_answer = ""

    return {
        "final_answer": final_answer,
        "answers": answers,
        "raw_answers": raw_answers
    }
    
def zero_shot_predict_deepseek(
        model, processor,train_dataset,test_dataset,
        indexes,
        query_idx,
        device="cuda",
        print_text=False,
        temperature=0.0,
        reps=1,
        system_message="",
        examples_message="",
        query_text_example="",
        query_text="",
        summary_after_examples_text="",
        max_new_tokens=512,
        image_first=False,
        all_labels=["FR-I","FR-II",],
    ):
    assert system_message != "" and query_text != ""
    
    imgq = test_dataset[query_idx]["image"]
    if imgq.mode != "RGB":
        imgq = imgq.convert("RGB")

    if image_first:
        conversation = [
            {
                "role": "<|User|>",
                "content": "<|ref|>" + system_message + "\n" + query_text + "<|/ref|><image>",
                "images": [imgq],            
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
    else:
        conversation = [
            {
                "role": "<|User|>",
                "content": "<|ref|>" + system_message + "<|/ref|><image>\n<|ref|>" + query_text + "<|/ref|>",
                "images": [imgq],            
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

    if print_text:
        print("Prompt:")
        print(conversation)

    prepare_inputs = processor(
        conversations=conversation,
        images=[imgq],
        force_batchify=True,
        system_prompt=system_message,
    ).to(device)
            
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    answers = Counter()
    raw_answers = []

    for i in range(reps):
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            temperature=temperature
        )
        
        raw = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        raw_answers.append(raw)
    
        if print_text:
            print(f"{prepare_inputs['sft_format'][0]}")
            print("Answer:")
            print(raw) 

        answer = extract_last_class(raw, all_labels)
        answers[answer] += 1
    if print_text:
        print(answers)
    final_answer = answers.most_common(1)[0][0]
    res = {}
    res["final_answer"] = final_answer
    res["answers"] = answers
    res["raw_answers"] = raw_answers
    
    return res

def zero_shot_predict_llava(
        model, processor,train_dataset,test_dataset,
        indexes,
        query_idx,
        device="cuda",
        print_text=False,
        temperature=0.0,
        reps=1,
        system_message="",
        examples_message="",
        query_text_example="",
        query_text="",
        summary_after_examples_text="",
        max_new_tokens=512,
        image_first=False,
        all_labels=["FR-I","FR-II",],
    ):
    imgq = test_dataset[query_idx]["image"]
        
    msgs = [
        {"role":"system",    "content":[{"type":"text","text":system_message}]},
    ]
    if image_first:
        msgs.extend([
            {"role":"user",      "content":[{"type":"image","image":imgq}, {"type":"text","text":query_text}]},
        ])
    else:
        msgs.extend([
            {"role":"user",      "content":[{"type":"text","text":query_text}, {"type":"image","image":imgq}]},
        ])

    
    text_input   = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(msgs)

    if print_text:
        print("Prompt:")
        print(text_input)

    inputs = processor(text=[text_input], images=image_inputs, return_tensors="pt").to(device)

    answers = Counter()
    raw_answers = []
    if temperature == 0:
        out_ids = model.generate(**inputs, 
                                     max_new_tokens=max_new_tokens,
                                     pad_token_id=processor.tokenizer.pad_token_id,
                                     num_return_sequences=reps,
                                    )[:, inputs.input_ids.shape[-1]:]
    else:
        out_ids = model.generate(**inputs, 
                                     max_new_tokens=max_new_tokens, 
                                     temperature=temperature, 
                                     do_sample=True,
                                     pad_token_id=processor.tokenizer.pad_token_id,
                                     num_return_sequences=reps,
                                     top_p=0.9, 
                                     top_k=50,
                                    )[:, inputs.input_ids.shape[-1]:]  
    raw_decoded  = processor.batch_decode(out_ids, skip_special_tokens=True)
    for raw in raw_decoded:
        raw = raw.strip()
        if print_text:
            print("Answer:")
            print(raw) 
        raw_answers.append(raw)
        answer = extract_last_class(raw, all_labels)
        answers[answer]+=1
    if print_text:
        print(answers)

    final_answer = answers.most_common(1)[0][0]
    
    res = {}
    res["final_answer"] = final_answer
    res["answers"] = answers
    res["raw_answers"] = raw_answers
    
    return res

def zero_shot_predict_gemini(
        client, model_name,train_dataset,test_dataset,
        indexes,
        query_idx,
        device="cuda",
        print_text=False,
        temperature=0.0,
        reps=1,
        system_message="",
        examples_message="",
        query_text_example="",
        query_text="",
        summary_after_examples_text="",
        max_new_tokens=2048,
        image_first=False,
        all_labels=["FR-I","FR-II",],
    ):
    assert system_message != "" and query_text != ""
    imgq = test_dataset[query_idx]["image"]
    imgq_bytes = PIL_to_bytes(imgq)

    if image_first:
        contents = [ 
            system_message, 
            types.Part.from_bytes(data=imgq_bytes, mime_type="image/png"), 
            query_text,
        ]
    else:
        contents = [ 
            system_message,
            query_text,
            types.Part.from_bytes(data=imgq_bytes, mime_type="image/png"), 
        ]
    
    answer = retry_generate_content(
        client.models,
        model=model_name,
        contents=contents,
        config=GenerateContentConfig(
            temperature=temperature,
            candidate_count=reps,
            seed=5,
            max_output_tokens=max_new_tokens,
        ),
    )
    answers = Counter()
    raw_answers = []
    for i, cand in enumerate(answer.candidates, start=1):
        try:
            raw = cand.content.parts[0].text
        except TypeError:
            print(cand)
            continue
        raw_answers.append(raw)
        if print_text:
            print(f"Candidate #{i} raw output:\n{raw}\n{'—'*20}")
    
        cls = extract_last_class(raw, all_labels)
        answers[cls] += 1
    if print_text:
        print(answer)
        print(answers)

    try:
        final_answer = answers.most_common(1)[0][0]
    except IndexError:
        final_answer = ""

    res = {}
    res["final_answer"] = final_answer
    res["answers"] = answers
    res["raw_answers"] = raw_answers
    
    return res

def few_shot_predict_deepseek(
        model, processor,train_dataset,test_dataset,
        indexes,
        query_idx,
        device="cuda",
        print_text=False,
        temperature=0.0,
        reps=1,
        system_message="",
        examples_message="",
        query_text_example="",
        query_text="",
        summary_after_examples_text="",
        max_new_tokens=512,
        image_first=False,
        all_labels=["FR-I","FR-II",],
    ):
    assert system_message != "" and query_text != ""
    
    imgq = test_dataset[query_idx]["image"].convert("RGB")
    
    content = system_message + examples_message

    class_counts = {"FR-I": 0, "FR-II": 0}
    images = []
    for idx in indexes:
        img_ex = train_dataset[idx]["image"].convert("RGB")
        lbl_ex = train_dataset[idx]["label"]
        class_counts[lbl_ex] += 1
        images.append(img_ex)
        content += f"<image>{query_text_example} {lbl_ex}"

    if image_first:
        content += "<image>" + query_text
    else:
        content += query_text + "<image>" 
    content += summary_after_examples_text

    images.append(imgq)

    conversation = [
        {
            "role": "<|User|>",
            "content": content,
            "images": images,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    if print_text:
        print("Prompt:")
        print(conversation)

    prepare_inputs = processor(
        conversations=conversation,
        images=[imgq],
        force_batchify=True,
        system_prompt=system_message,
    ).to(device)
            
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    answers = Counter()
    raw_answers = []

    for i in range(reps):
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            temperature=temperature
        )
        
        raw = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        raw_answers.append(raw)
    
        if print_text:
            print(f"{prepare_inputs['sft_format'][0]}")
            print("Answer:")
            print(raw) 

        answer = extract_last_class(raw, all_labels)
        answers[answer] += 1
    if print_text:
        print(answers)
    final_answer = answers.most_common(1)[0][0]
    res = {}
    res["final_answer"] = final_answer
    res["answers"] = answers
    res["raw_answers"] = raw_answers
    
    return res

def _is_llava16(model):
    model_name = model.__class__.__name__.lower()
    config_type = getattr(getattr(model, "config", None), "model_type", "").lower()
    is_llava16 = ("llavanext" in model_name) or ("llava" in config_type) or ("llava" in model_name)
    return is_llava16

def normalize_messages_for_model(msgs, model):
    model_name = model.__class__.__name__.lower()
    config_type = getattr(getattr(model, "config", None), "model_type", "").lower()
    is_llava16 = ("llavanext" in model_name) or ("llava" in config_type) or ("llava" in model_name)

    images, normalized = [], []

    for m in msgs:
        role = m.get("role", "")
        content = m.get("content")

        if not isinstance(content, list):
            if is_llava16:
                normalized.append({"role": role, "content": [{"type": "text", "text": str(content)}]})
            else:
                normalized.append({"role": role, "content": content})
            continue

        if is_llava16:
            if role in {"system", "assistant"}:
                chunks = []
                for it in content:
                    if it.get("type") == "text":
                        chunks.append({"type": "text", "text": it.get("text", "")})
                if not chunks:
                    chunks = [{"type": "text", "text": ""}]
                normalized.append({"role": role, "content": chunks})
            else:
                chunks = []
                for it in content:
                    t = it.get("type")
                    if t == "image":
                        chunks.append({"type": "image"})
                        img = it.get("image")
                        if img is not None:
                            images.append(img)
                    elif t == "text":
                        chunks.append({"type": "text", "text": it.get("text", "")})
                normalized.append({"role": role, "content": chunks})
        else:
            chunks = []
            for it in content:
                t = it.get("type")
                if t == "image":
                    img = it.get("image")
                    if img is not None:
                        images.append(img)
                    chunks.append(it)
                elif t == "text":
                    chunks.append({"type": "text", "text": it.get("text", "")})
            normalized.append({"role": role, "content": chunks})

    return normalized, images


def few_shot_predict_llava(
        model, processor,train_dataset,test_dataset,
        indexes,
        query_idx,
        device="cuda",
        print_text=False,
        temperature=0.0,
        reps=1,
        system_message="",
        examples_message="",
        query_text_example="",
        query_text="",
        summary_after_examples_text="",
        max_new_tokens=512,
        image_first=False,
        all_labels=["FR-I","FR-II",],
    ):
    assert system_message != "" and query_text != ""
    imgq = test_dataset[query_idx]["image"]
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": system_message+examples_message}]},
    ]
    classes_seen = ""
    for n, idx in enumerate(indexes, start=1):
        img_ex = train_dataset[idx]["image"]
        lbl_ex = train_dataset[idx]["label"]
        cur_answer = lbl_ex
        classes_seen += str(train_dataset[idx]["label"]) + " "
        msgs.append({
            "role": "user",
            "content": [
                
                {"type": "text",  "text": (
                    query_text_example
                )},
                {"type": "image", "image": img_ex},
            ]
        })
        msgs.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": (
                    f"{cur_answer}"
                )}
            ]
        })


    if image_first:
        msgs.extend([
            {"role":"user",      "content":[{"type":"image","image":imgq}, {"type":"text","text":query_text}]},
        ])
    else:
        msgs.extend([
            {"role":"user",      "content":[{"type":"text","text":query_text}, {"type":"image","image":imgq}]},
        ])
    msgs, image_inputs = normalize_messages_for_model(msgs, model)

    text_input = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    if not _is_llava16(model):
        image_inputs, _ = process_vision_info(msgs)
    if print_text:
        print("Msgs:")
        print(msgs)
        print("image_inputs:")
        print(len(image_inputs))
        print("Prompt:")
        print(text_input)

    inputs = processor(
        text=[text_input], images=image_inputs,
        return_tensors="pt"
    ).to(device)

    answers = Counter()
    raw_answers = []
    if temperature == 0:
        out_ids = model.generate(**inputs, 
                                     max_new_tokens=max_new_tokens,
                                     pad_token_id=processor.tokenizer.pad_token_id,
                                     num_return_sequences=reps,
                                    )[:, inputs.input_ids.shape[-1]:]
    else:
        out_ids = model.generate(**inputs, 
                                     max_new_tokens=max_new_tokens, 
                                     temperature=temperature, 
                                     do_sample=True,
                                     pad_token_id=processor.tokenizer.pad_token_id,
                                     num_return_sequences=reps,
                                     top_p=0.9, 
                                     top_k=50,
                                    )[:, inputs.input_ids.shape[-1]:]  
    raw_decoded  = processor.batch_decode(out_ids, skip_special_tokens=True)
    for raw in raw_decoded:
        raw = raw.strip()
        if print_text:
            print("Answer:")
            print(raw) 
        raw_answers.append(raw)
        answer = extract_last_class(raw, all_labels)
        answers[answer]+=1
    if print_text:
        print(answers)

    final_answer = answers.most_common(1)[0][0]
    
    res = {}
    res["final_answer"] = final_answer
    res["answers"] = answers
    res["raw_answers"] = raw_answers
    
    return res

def few_shot_predict_gemini(
        client, model_name,train_dataset,test_dataset,
        indexes,
        query_idx,
        device="cuda",
        print_text=False,
        temperature=0.0,
        reps=1,
        system_message="",
        examples_message="",
        query_text_example="",
        query_text="",
        summary_after_examples_text="",
        max_new_tokens=512,
        image_first=False,
        all_labels=["FR-I","FR-II",],
    ):
    assert system_message != "" and query_text != ""
    imgq = test_dataset[query_idx]["image"]
    imgq_bytes = PIL_to_bytes(imgq)
    
    contents = [ 
        system_message,
        examples_message
    ]

    for idx in indexes:
        img = train_dataset[idx]["image"]
        ans = train_dataset[idx]["label"] 

        img_bytes = PIL_to_bytes(img)
        contents.extend(
                    [
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        query_text_example,
                        ans,
                        
                    ]
                   )
    if image_first:
        contents.extend(
                        [
                            summary_after_examples_text,
                            types.Part.from_bytes(data=imgq_bytes, mime_type="image/png"),
                            query_text,
                        ]
                       )
    else:
        contents.extend(
                        [
                            summary_after_examples_text,
                            query_text,
                            types.Part.from_bytes(data=imgq_bytes, mime_type="image/png"),
                            
                        ]
                       )

    answer = retry_generate_content(
        client.models,
        model=model_name,
        contents=contents,
        config=GenerateContentConfig(
            temperature=temperature,
            candidate_count=reps,
            seed=5,
            max_output_tokens=max_new_tokens,
        ),
    )


    if print_text:
        print("Input contents:")
        print(contents)
    time.sleep(GEMINI_SLEEP)
    answers = Counter()
    raw_answers = []
    for i, cand in enumerate(answer.candidates, start=1):
        try:
            raw = cand.content.parts[0].text
        except TypeError:
            print(cand)
            continue
        raw_answers.append(raw)
        if print_text:
            print(f"Candidate #{i} raw output:\n{raw}\n{'—'*20}")
    
        cls = extract_last_class(raw, all_labels)
        answers[cls] += 1
    if print_text:
        print(raw_answers)
        print(answers)

    try:
        final_answer = answers.most_common(1)[0][0]
    except IndexError:
        final_answer = ""

    res = {}
    res["final_answer"] = final_answer
    res["answers"] = answers
    res["raw_answers"] = raw_answers
    
    return res

def few_shot_predict_openai(
        client, model_name, train_dataset, test_dataset,
        indexes,
        query_idx,
        device="cuda",
        print_text=False,
        temperature=0.0,
        reps=1,
        system_message="",
        examples_message="",
        query_text_example="",
        query_text="",
        summary_after_examples_text="",
        max_new_tokens=512,
        image_first=False,
        all_labels=["FR-I","FR-II",],
    ):
    assert system_message != "" and query_text != ""

    imgq = test_dataset[query_idx]["image"]
    base64_imgq = PIL_to_base64(imgq)

    user_content = []

    if examples_message:
        user_content.append({"type": "text", "text": examples_message})

    for idx in indexes:
        img = train_dataset[idx]["image"]
        ans = train_dataset[idx]["label"]
        base64_img = PIL_to_base64(img)

        user_content.extend([
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_img}",
                    "detail": "auto"
                }
            },
            {"type": "text", "text": query_text_example},
            {"type": "text", "text": ans}
        ])

    if summary_after_examples_text:
        user_content.append({"type": "text", "text": summary_after_examples_text})

    if image_first:
        user_content.extend([
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_imgq}",
                    "detail": "auto"
                }
            },
            {"type": "text", "text": query_text}
        ])
    else:
        user_content.extend([
            {"type": "text", "text": query_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_imgq}",
                    "detail": "auto"
                }
            },
        ])  

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]

    response = chat_with_retries(
        client=client, model=model_name, messages=messages, max_tokens=max_new_tokens, n=reps, temperature=temperature
    )

    if print_text:
        print("Input contents:")
        print(messages)

    answers = Counter()
    raw_answers = []

    for i, choice in enumerate(response.choices, start=1):
        raw = choice.message.content.strip()
        raw_answers.append(raw)
        if print_text:
            print(f"Candidate #{i} raw output:\n{raw}\n{'—'*20}")

        cls = extract_last_class(raw, all_labels)
        answers[cls] += 1

    try:
        final_answer = answers.most_common(1)[0][0]
    except IndexError:
        final_answer = ""

    return {
        "final_answer": final_answer,
        "answers": answers,
        "raw_answers": raw_answers
    }
    
def theory_shot_predict_deepseek(
        model, processor,train_dataset,test_dataset,
        indexes,
        query_idx,
        device="cuda",
        print_text=False,
        temperature=0.0,
        reps=1,
        system_message="",
        examples_message="",
        query_text_example="",
        query_text="",
        summary_after_examples_text="",
        max_new_tokens=512,
        image_first=False,
        all_labels=["FR-I","FR-II",],
    ):
    imgq = test_dataset[query_idx]["image"]
    if imgq.mode != "RGB":
        imgq = imgq.convert("RGB")
    images = [imgq]

    diagram_img = PIL.Image.open(diagram_path).convert("RGB")
    diagram_text = "<image><|ref|> First image. This is the diagram showing the patterns of FR-I and FR-II classes. <|/ref|>\n"
    images.insert(0, diagram_img)

    if image_first:
        conversation = [
            {
                "role": "<|User|>",
                "content": diagram_text + "<|ref|>" + system_message + "<|/ref|><image>\n<|ref|> Second image. " + query_text + "<|/ref|>",
                "images": images,                  
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
    else:
        conversation = [
            {
                "role": "<|User|>",
                "content": diagram_text + "<|ref|>" + system_message + "<|/ref|>\n<|ref|> Second image. " + query_text + "<|/ref|><image>",
                "images": images,                  
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

    if print_text:
        print("Prompt:")
        print(conversation)

    prepare_inputs = processor(
        conversations=conversation,
        images=images,
        force_batchify=True,
        system_prompt=system_message,
    ).to(device)
            
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    raw_answers = []
    answers = {"FR-I" : 0, "FR-II": 0}

    for i in range(reps):
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            temperature=temperature
        )
        
        raw = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        raw_answers.append(raw)
    
        if print_text:
            print(f"{prepare_inputs['sft_format'][0]}")
            print("Answer:")
            print(raw) 

        answer = extract_last_class(raw, all_labels)
        answers[answer] += 1
    if print_text:
        print(answers)
    final_answer = answers.most_common(1)[0][0]
    res = {}
    res["final_answer"] = final_answer
    res["answers"] = answers
    res["raw_answers"] = raw_answers
    
    return res

def theory_shot_predict_llava(
        model, processor,train_dataset,test_dataset,
        indexes,
        query_idx,
        device="cuda",
        print_text=False,
        temperature=0.0,
        reps=1,
        system_message="",
        examples_message="",
        query_text_example="",
        query_text="",
        summary_after_examples_text="",
        max_new_tokens=512,
        image_first=False,
        all_labels=["FR-I","FR-II",],
    ):
    imgq = test_dataset[query_idx]["image"]
    cartoon_img = PIL.Image.open(DIAGRAM_PATH)

    msgs = [
        {"role":"system",    "content":[{"type":"image", "image":cartoon_img},{"type":"text","text":system_message}]},
    ]
    if image_first:
        msgs.extend([
            {"role":"user",      "content":[{"type":"image","image":imgq}, {"type":"text","text":query_text}]},
        ])
    else:
        msgs.extend([
            {"role":"user",      "content":[{"type":"text","text":query_text}, {"type":"image","image":imgq}]},
        ])
    print("before:", msgs)

    msgs, image_inputs = normalize_messages_for_model(msgs, model)
    print("after:", msgs)
    text_input   = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if not _is_llava16(model):
        image_inputs, _ = process_vision_info(msgs)
    if print_text:
        print("Prompt:")
        print(text_input)

    inputs = processor(text=[text_input], images=image_inputs, return_tensors="pt").to(device)
    
    answers = Counter()
    raw_answers = []
    if temperature == 0:
        out_ids = model.generate(**inputs, 
                                     max_new_tokens=max_new_tokens,
                                     pad_token_id=processor.tokenizer.pad_token_id,
                                     num_return_sequences=reps,
                                    )[:, inputs.input_ids.shape[-1]:]
    else:
        out_ids = model.generate(**inputs, 
                                     max_new_tokens=max_new_tokens, 
                                     temperature=temperature, 
                                     do_sample=True,
                                     pad_token_id=processor.tokenizer.pad_token_id,
                                     num_return_sequences=reps,
                                     top_p=0.9, 
                                     top_k=50,
                                    )[:, inputs.input_ids.shape[-1]:]  
    raw_decoded  = processor.batch_decode(out_ids, skip_special_tokens=True)
    for raw in raw_decoded:
        raw = raw.strip()
        if print_text:
            print("Answer:")
            print(raw) 
        raw_answers.append(raw)
        answer = extract_last_class(raw, all_labels)
        answers[answer]+=1
    if print_text:
        print(answers)

    final_answer = answers.most_common(1)[0][0]
    
    res = {}
    res["final_answer"] = final_answer
    res["answers"] = answers
    res["raw_answers"] = raw_answers
    
    return res

def theory_shot_predict_gemini(
        model, processor,train_dataset,test_dataset,
        indexes,
        query_idx,
        device="cuda",
        print_text=False,
        temperature=0.0,
        reps=1,
        system_message="",
        examples_message="",
        query_text_example="",
        query_text="",
        summary_after_examples_text="",
        max_new_tokens=512,
        image_first=False,
        all_labels=["FR-I","FR-II",],
    ):
    imgq = test_dataset[query_idx]["image"]
    imgq_bytes = PIL_to_bytes(imgq)

    cartoon_img = PIL.Image.open(DIAGRAM_PATH)
    cartoon_bytes = PIL_to_bytes(cartoon_img)

    if image_first:
        contents = [ 
            types.Part.from_bytes(data=cartoon_bytes, mime_type="image/png"), 
            system_message, 
            types.Part.from_bytes(data=imgq_bytes, mime_type="image/png"),
            query_text,
        ]
    else:
        contents = [ 
            types.Part.from_bytes(data=cartoon_bytes, mime_type="image/png"), 
            system_message, 
            query_text,
            types.Part.from_bytes(data=imgq_bytes, mime_type="image/png"),
        ]

    answer = retry_generate_content(
        model.models,
        model=processor,
        contents=contents,
        config=GenerateContentConfig(
            temperature=temperature,
            candidate_count=reps,
            seed=5,
            max_output_tokens=max_new_tokens,
        ),
    )

    answers = Counter()
    time.sleep(GEMINI_SLEEP)
    raw_answers = []
    for i, cand in enumerate(answer.candidates, start=1):
        try:
            raw = cand.content.parts[0].text
        except TypeError:
            print(cand)
            continue
        raw_answers.append(raw)
        if print_text:
            print(f"Candidate #{i} raw output:\n{raw}\n{'—'*20}")
    
        cls = extract_last_class(raw, all_labels)
        answers[cls] += 1
    if print_text:
        print(answer)
        print(answers)

    try:
        final_answer = answers.most_common(1)[0][0]
    except IndexError:
        final_answer = ""
    print(final_answer, raw_answers)
    res = {}
    res["final_answer"] = final_answer
    res["answers"] = answers
    res["raw_answers"] = raw_answers
    return res

def majority_vote(train_dataset, indices, label_key="label",
                  tie_break="lex", return_info=True, rng=None):
    """
    tie_break: "lex" (default: smallest label string),
               "first" (first seen among indices),
               "random" (uniform among tied labels)
    """
    idxs = list(indices)
    if not idxs:
        raise ValueError("indices is empty")

    labels = [train_dataset[i][label_key] for i in idxs]
    counts = Counter(labels)
    maxc = max(counts.values())
    winners = [lab for lab, c in counts.items() if c == maxc]

    if len(winners) == 1:
        winner = winners[0]
    else:
        if tie_break == "first":
            for lab in labels:
                if lab in winners:
                    winner = lab
                    break
        elif tie_break == "random":
            if rng is None:
                winner = random.choice(winners)
            else:
                winner = rng.choice(winners)
        else:
            winner = sorted(winners)[0]

    if return_info:
        vote_share = counts[winner] / len(labels)
        return winner, vote_share, dict(counts)
    return winner

def knn_shot(
        client, model_name, train_dataset,test_dataset,
        indexes,
        query_idx,
        device="cuda",
        print_text=False,
        temperature=0.0,
        reps=1,
        system_message="",
        examples_message="",
        query_text_example="",
        query_text="",
        summary_after_examples_text="",
        max_new_tokens=512,
        image_first=False,
    ):
    res = {}
    res["final_answer"] =  majority_vote(train_dataset, indexes, return_info=False)
    return res

def theory_shot_predict_openai(
        client, model_name, train_dataset,test_dataset,
        indexes,
        query_idx,
        device="cuda",
        print_text=False,
        temperature=0.0,
        reps=1,
        system_message="",
        examples_message="",
        query_text_example="",
        query_text="",
        summary_after_examples_text="",
        max_new_tokens=512,
        image_first=False,
        all_labels=["FR-I","FR-II",],
    ):
    assert system_message != "" and query_text != ""

    imgq = test_dataset[query_idx]["image"]
    base64_imgq = PIL_to_base64(imgq)

    cartoon_img = PIL.Image.open(DIAGRAM_PATH)
    base64_cartoon = PIL_to_base64(cartoon_img)
        
    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_cartoon}",
                        "detail": "auto"
                    }
                },
                {"type": "text", "text": query_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_imgq}",
                        "detail": "auto"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        n=reps,
        max_tokens=max_new_tokens
    )

    answers = Counter()
    raw_answers = []

    for i, choice in enumerate(response.choices, start=1):
        raw = choice.message.content.strip()
        raw_answers.append(raw)
        if print_text:
            print(f"Candidate #{i} raw output:\n{raw}\n{'—'*20}")

        cls = extract_last_class(raw, all_labels)
        answers[cls] += 1

    try:
        final_answer = answers.most_common(1)[0][0]
    except IndexError:
        final_answer = ""

    res = {
        "final_answer": final_answer,
        "answers": answers,
        "raw_answers": raw_answers
    }

    return res
    
def zero_shot_predict_func(model_name):
    if "deepseek" in model_name:
        return zero_shot_predict_deepseek
    elif model_name in [
        "Qwen/Qwen2-VL-7B-Instruct",
        "llava-hf/llava-1.5-7b-hf",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "Qwen/Qwen3-8B",
    ] or "Qwen2" in model_name:
        return zero_shot_predict_llava
    elif "gemini" in model_name:
        return zero_shot_predict_gemini
    elif "gpt" in model_name:
        return zero_shot_predict_openai
    else:
        raise NotImplementedError()

def theory_shot_predict_func(model_name):
    if "deepseek" in model_name:
        return theory_shot_predict_deepseek
    elif "gpt" in model_name:
        return theory_shot_predict_openai
    elif model_name in [
        "Qwen/Qwen2-VL-7B-Instruct",
        "llava-hf/llava-1.5-7b-hf",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "Qwen/Qwen3-8B",
    ] or "Qwen2" in model_name:
        return theory_shot_predict_llava
    elif "gemini" in model_name:
        return theory_shot_predict_gemini
    else:
        raise NotImplementedError()

def few_shot_predict_func(model_name):
    if "deepseek" in model_name:
        return few_shot_predict_deepseek
    elif "gpt" in model_name:
        return few_shot_predict_openai
    elif model_name in [
        "Qwen/Qwen2-VL-7B-Instruct",
        "llava-hf/llava-1.5-7b-hf",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "Qwen/Qwen3-8B",
        "HuggingFaceM4/idefics2-8b-chatty",
        "OpenGVLab/InternVL2-1B",
        "openbmb/MiniCPM-V-2_6",
        "microsoft/Phi-3.5-vision-instruct",
    ] or "Qwen2" in model_name:
        
        return few_shot_predict_llava
    elif "gemini" in model_name:
        return few_shot_predict_gemini
    elif "knn" in model_name:
        return knn_shot
    else:
        raise NotImplementedError()
