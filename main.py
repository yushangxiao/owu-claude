"""
title: Anthropic Manifold Pipe
authors: coker
author_url: https://linux.do/u/coker/summary
based: https://openwebui.com/f/justinrahb/anthropic
version: 0.0.2
"""

import os
import requests
import json
import time
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message
import re


class Pipe:
    class Valves(BaseModel):
        ANTHROPIC_API_URL: str = Field(default="https://api.anthropic.com")
        ANTHROPIC_API_KEY: str = Field(default="")
        THINKING_TOKENS: int = Field(default=32000)

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "Anthropic: "
        self.valves = self.Valves()
        self.enable_thinking_models = ["claude-3-7-sonnet-20250219"]
        self.MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image

        self.is_thinking = False
        pass

    def get_anthropic_models(self):
        res = [
            {"id": "claude-3-7-sonnet-20250219", "name": "claude-3-7-sonnet-20250219"},
            {"id": "claude-3-5-sonnet-latest", "name": "claude-3.5-sonnet-latest"},
        ]
        if self.enable_thinking_models:
            res.extend(
                [
                    {
                        "id": f"{model}-think",
                        "name": f"{model}-think",
                    }
                    for model in self.enable_thinking_models
                ]
            )
        return res

    def pipes(self) -> List[dict]:
        return self.get_anthropic_models()

    def process_image(self, image_data):
        """Process image data with size validation."""
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]

            # Check base64 image size
            image_size = len(base64_data) * 3 / 4  # Convert base64 size to bytes
            if image_size > self.MAX_IMAGE_SIZE:
                raise ValueError(
                    f"Image size exceeds 5MB limit: {image_size / (1024 * 1024):.2f}MB"
                )

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            # For URL images, perform size check after fetching
            url = image_data["image_url"]["url"]
            response = requests.head(url, allow_redirects=True)
            content_length = int(response.headers.get("content-length", 0))

            if content_length > self.MAX_IMAGE_SIZE:
                raise ValueError(
                    f"Image at URL exceeds 5MB limit: {content_length / (1024 * 1024):.2f}MB"
                )

            return {
                "type": "image",
                "source": {"type": "url", "url": url},
            }

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        system_message, messages = pop_system_message(body["messages"])

        processed_messages = []
        total_image_size = 0

        for message in messages:
            processed_content = []
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item["type"] == "text":
                        content = item["text"]
                        if "<think>" in content and "</think>" in content:
                            content = re.sub(r"<think>.*?</think>", "", content)
                        processed_content.append({"type": "text", "text": content})
                    elif item["type"] == "image_url":
                        processed_image = self.process_image(item)
                        processed_content.append(processed_image)

                        # Track total size for base64 images
                        if processed_image["source"]["type"] == "base64":
                            image_size = len(processed_image["source"]["data"]) * 3 / 4
                            total_image_size += image_size
                            if (
                                total_image_size > 100 * 1024 * 1024
                            ):  # 100MB total limit
                                raise ValueError(
                                    "Total size of images exceeds 100 MB limit"
                                )
            else:
                content = message.get("content", "")
                if "<think>" in content and "</think>" in content:
                    # 移除思考内容
                    content = re.sub(r"<think>.*?</think>", "", content)
                processed_content = [{"type": "text", "text": content}]

            processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )
        model_id = body["model"][body["model"].find(".") + 1 :]
        payload = {
            # "model": ,
            "messages": processed_messages,
            "max_tokens": body.get("max_tokens", 128000),
            "temperature": body.get("temperature", 0.8),
            "top_k": body.get("top_k", 40),
            "top_p": body.get("top_p", 0.9),
            "stop_sequences": body.get("stop", []),
            **({"system": str(system_message)} if system_message else {}),
            "stream": body.get("stream", False),
        }
        if model_id.endswith("-think"):
            model_id = model_id[: model_id.find("-think")]
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.valves.THINKING_TOKENS,
            }
            payload["temperature"] = 1.0
            payload.pop("top_k", None)
            payload.pop("top_p", None)
        if payload.get("max_tokens", 128000) > 8192 and not "3-7" in model_id:
            payload["max_tokens"] = 8192
        payload["model"] = model_id
        headers = {
            "x-api-key": self.valves.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "anthropic-beta": "output-128k-2025-02-19",
        }

        url = self.valves.ANTHROPIC_API_URL + "/v1/messages"

        try:
            if body.get("stream", False):
                return self.stream_response(url, headers, payload)
            else:
                return self.non_stream_response(url, headers, payload)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(self, url, headers, payload):
        try:
            with requests.post(
                url, headers=headers, json=payload, stream=True, timeout=(3.05, 60)
            ) as response:
                if response.status_code != 200:
                    raise Exception(
                        f"HTTP Error {response.status_code}: {response.text}"
                    )

                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data["type"] == "content_block_start":
                                    if data["content_block"]["type"] == "thinking":
                                        self.is_thinking = True
                                        yield "<think>\n"
                                    if "text" in data["content_block"]:
                                        yield data["content_block"]["text"]
                                elif data["type"] == "content_block_delta":
                                    if self.is_thinking and "thinking" in data["delta"]:
                                        yield data["delta"]["thinking"]
                                    if "text" in data["delta"]:
                                        yield data["delta"]["text"]
                                elif data["type"] == "message_stop":
                                    break
                                elif data["type"] == "message":
                                    for content in data.get("content", []):
                                        if content["type"] == "text":
                                            yield content["text"]
                                elif (
                                    data["type"] == "content_block_stop"
                                    and self.is_thinking
                                ):
                                    self.is_thinking = False
                                    yield "</think>\n"
                                time.sleep(
                                    0.01
                                )  # Delay to avoid overwhelming the client

                            except json.JSONDecodeError:
                                print(f"Failed to parse JSON: {line}")
                            except KeyError as e:
                                print(f"Unexpected data structure: {e}")
                                print(f"Full data: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            yield f"Error: Request failed: {e}"
        except Exception as e:
            print(f"General error in stream_response method: {e}")
            yield f"Error: {e}"

    def non_stream_response(self, url, headers, payload):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=(3.05, 60)
            )
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")

            res = response.json()
            return (
                res["content"][0]["text"] if "content" in res and res["content"] else ""
            )
        except requests.exceptions.RequestException as e:
            print(f"Failed non-stream request: {e}")
            return f"Error: {e}"
