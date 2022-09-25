import glob
import json
import os
import re
import traceback

import requests

if __name__ == "__main__":
    target_locale = "en"
    re_break = re.compile(r"\n")
    re_end_break = re.compile(r"\\n$")
    re_message = re.compile(r'"(.*)"')

    messages = []
    with open("i18n/messages.pot", mode="r", encoding="utf-8") as f:
        messages = f.readlines()

    for file_path in glob.glob(os.path.join("i18n\\**\\messages.po"), recursive=True):
        print(file_path)
        is_ja = "ja" in file_path

        try:
            trans_messages = []
            with open(file_path, mode="r", encoding="utf-8") as f:
                trans_messages = f.readlines()

            msg_id = None
            for i, org_msg in enumerate(messages):
                if i < 18:
                    # ヘッダはそのまま
                    continue

                if "msgid" in org_msg:
                    m = re_message.search(org_msg)
                    if m:
                        msg_id = m.group()
                        continue

                if msg_id and "msgstr" in org_msg:
                    transed_msg_idxs = [
                        n + 1
                        for n, msg in enumerate(trans_messages)
                        if "msgid" in msg
                        and "msgstr" in trans_messages[n + 1]
                        and msg_id in msg
                        and '""' not in trans_messages[n + 1]
                    ]

                    msg_str = msg_id
                    if transed_msg_idxs:
                        # 既に翻訳済みの場合、記載されてる翻訳情報を転載
                        messages[i] = trans_messages[transed_msg_idxs[0]]
                    else:
                        if is_ja:
                            messages[i] = f"msgstr {msg_id}"
                        else:
                            # 値がないメッセージを翻訳
                            params = (
                                ("text", msg_id),
                                ("source", "ja"),
                                ("target", target_locale),
                            )

                            # GASを叩く
                            # https://qiita.com/satto_sann/items/be4177360a0bc3691fdf
                            response = requests.get(
                                "https://script.google.com/macros/s/AKfycbzZtvOvf14TaMdRIYzocRcf3mktzGgXvlFvyczo/exec",
                                params=params,
                            )

                            # 結果を解析
                            results = json.loads(response.text)

                            if "text" in results:
                                v = results["text"]
                                messages[i] = f"msgstr {v}"
                                print(f"「{msg_id}」 -> 「{v}」")
                        msg_id = None

            for i, message in enumerate(messages):
                message = re_break.sub("\\\\n", message)
                message = re_end_break.sub("\\n", message)
                messages[i] = message

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(messages)
        except Exception as e:
            print("*** Message Translate ERROR ***\n%s", traceback.format_exc())
