import argparse, os, re, gc
import joblib
import pandas as pd
import numpy as np
import torch
import lightgbm as lgb
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoTokenizer


_SHARED_MODELS_DIR = os.environ.get("SHARED_MODELS_PATH", "/shared_models")
MODEL_EMBED_PATH = os.path.join(_SHARED_MODELS_DIR, "Qwen/Qwen3-VL-Embedding-2B")
MODEL_LLM_PATH   = os.path.join(_SHARED_MODELS_DIR, "Qwen/Qwen3.5-4B")

ARTIFACT_PATH = "classifiers.pkl"

RULES = {
    "БАД": "Бан: если нет маркировки БАД или это спортпит. Не бан: если есть прямое указание на БАД.",
    "Легковоспламеняющиеся": "Бан: если это источник огня или горючее. Не бан: если устройство без топлива или уголь как компонент."
}

PATTERNS = [
    r'\bбад\b|\bбады\b|\bбадов\b|\bбадами\b',
    r'биологически активн\w* добавк\w*',
    r'dietary supplement',
    r'пищев\w* добавк\w*',
    r'пробиотик|пробиотическ|биотин|омега|коллаген|рыбий жир',
    r'витамин|минерал|экстракт|пищев\w* волокон',
    r'аминокислот|bcaa|протеин|сывороточн|карнитин|креатин|гейнер|предтрен|спортивн\w* питан',
    r'не является бад|не биодобавк|не является биологически',
    r'спичк|зажигалк|огнив|керосин',
    r'горюче|топлив|бензин|газ|пропан|бутан|ацетон|спирт|розжиг',
    r'аэрозол|спрей|баллон',
    r'мангал|гриль|барбекю|плита|духовк|камин|печь|горелк',
    r'уголь\w*.{0,30}(рисован|фильтр|кальян)|активированн\w* уголь',
    r'без содержим|без газа|без топлива|без баллона|пустой',
]


def rule_features(texts):
    F = np.zeros((len(texts), len(PATTERNS)), dtype=np.float32)
    for i, t in enumerate(texts):
        for j, p in enumerate(PATTERNS):
            F[i, j] = int(bool(re.search(p, t)))

    mark  = np.clip(F[:,0] + F[:,1] + F[:,2], 0, 1)
    sport = F[:,6]
    neg   = F[:,7]
    supp  = np.clip(F[:,4] + F[:,5], 0, 1)
    food  = F[:,3]

    G = np.column_stack([
        F,
        mark * sport,
        mark * (1-sport) * (1-neg),
        (1-mark) * supp,
        (1-mark) * food,
        sport,
        neg,
    ]).astype(np.float32)
    return G


def embed_texts_batch(texts, model, processor, batch_size=128):
    all_embs = []
    device = model.device
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch = [t if isinstance(t, str) and t.strip() else "empty description" for t in batch]
        inputs = processor(text=batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        sum_emb = torch.sum(hidden * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        emb = (sum_emb / sum_mask).cpu().numpy().astype(np.float32)
        all_embs.append(emb)
    return np.vstack(all_embs)


def predict_verdict(test_df, X_emb, artifact):
    preds = np.zeros(len(test_df), dtype=int)   # страховка от мусора
    for cat in test_df['category'].unique():
        cat = str(cat)                 # гарантированно строка
        if cat not in artifact:        # страховка от неизвестной категории
            preds[test_df['category'] == cat] = 1
            continue
        mask = test_df['category'] == cat
        if mask.sum() == 0:
            continue
        names = test_df.loc[mask, 'name'].values
        descriptions = test_df.loc[mask, 'description'].values
        texts = (pd.Series(names).fillna('') + ' ' + pd.Series(descriptions).fillna('')).str.lower().values
        X = np.hstack([X_emb[mask], rule_features(texts)])
        a = artifact[cat]
        variant = a.get("variant", "ens")   # на случай, если ключа нет
        if variant == "lr":
            p = a['lr'].predict_proba(X)[:, 1]
        elif variant == "lgb":
            p = a['lgb'].predict_proba(X)[:, 1]
        else:   # ens
            p = 0.5 * (a['lr'].predict_proba(X)[:, 1] + a['lgb'].predict_proba(X)[:, 1])
        preds[mask] = (p >= a['thr']).astype(int)
    return preds


def generate_explanations_batch(texts, categories, verdicts, tokenizer, model, batch_size=64):
    results = []
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    i, n = 0, len(texts)
    bs = batch_size
    while i < n:
        idxs = list(range(i, min(i + bs, n)))
        prompts = []
        for j in idxs:
            rule = RULES.get(categories[j], "")
            verdict_text = "бан" if verdicts[j] == 0 else "не бан"
            prompts.append(f"""Ты - модератор. Объясни вердикт для товара.
Категория: {categories[j]}
Правило: {rule}
Товар: {texts[j][:300]}
Вердикт: {verdict_text}

Формат ответа СТРОГО: <комментарий>КРАТКОЕ ОБОСНОВАНИЕ (2-3 предложения, 60-200 символов)<вердикт>{verdict_text}
Ответ:""")
        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            decoded = tokenizer.batch_decode(
                generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            for raw, j in zip(decoded, idxs):
                verdict_text = "бан" if verdicts[j] == 0 else "не бан"
                #постобработка с добавлением тега, если его нет
                clean = re.sub(r'<[^>]*>', '', raw).strip()
                
                #комментарий между тегами
                match = re.search(r'<комментарий>(.*?)<вердикт>', raw, re.DOTALL | re.IGNORECASE)
                if match:
                    comment = match.group(1).strip()
                else:
                    # Тега <вердикт> нет — модель не успела его сгенерировать
                    comment = clean
                    if comment.endswith(verdict_text):
                        comment = comment[:-len(verdict_text)].strip()
                
                comment = re.sub(r'\s+', ' ', comment)
                
                # Длина 50-300
                if len(comment) < 50:
                    comment = (comment + " Товар проверен на соответствие правилам категории.").strip()
                if len(comment) > 300:
                    comment = comment[:297] + "..."
                
                # ВСЕГДА добавляем тег <вердикт>, даже если модель забыла
                results.append(f"<комментарий>{comment}<вердикт>{verdict_text}")
            
            i += len(idxs)
            bs = min(batch_size, bs * 2)
            torch.cuda.empty_cache()
            print(f"  [LLM] {i}/{n}", flush=True)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            print(f"  [LLM] OOM: батч -> {bs}", flush=True)
    return results


def generate_answer_rule(row, verdict):
    name = row['name']
    desc = row['description'] if isinstance(row['description'], str) else ''
    text = f"{name} {desc}".lower()
    category = str(row['category'])
    verdict_text = "бан" if verdict == 0 else "не бан"

    if category == 'БАД':
        if any(w in text for w in ['бад', 'биодобавка', 'dietary supplement']) and \
           not any(w in text for w in ['протеин', 'bcaa', 'аминокислот']):
            reason = "Товар содержит указание на БАД."
        else:
            reason = "Товар не является БАД."
    else:
        if any(w in text for w in ['спички', 'зажигалка', 'горючее', 'топливо']):
            reason = "Обнаружены признаки горючего вещества."
        elif any(w in text for w in ['мангал', 'гриль', 'плита']) and 'без' not in text:
            reason = "Устройство для огня, возможно, с горючим."
        else:
            reason = "Признаки легковоспламеняющихся отсутствуют."

    comment = reason
    if len(comment) < 50:
        comment += " Дополнительная проверка не требуется."
    if len(comment) > 300:
        comment = comment[:297] + "..."
    return f"<комментарий>{comment}<вердикт>{verdict_text}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', '--test-data-path', '-i', dest='test_data_path', type=str, required=True, help='путь к test.csv')
    parser.add_argument('--output-path', '--output_path', '-o', dest='output_path', type=str, required=True, help='путь к выходному файлу')
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_data_path)
    test_df['text'] = test_df['name'].fillna('') + ' ' + test_df['description'].fillna('') + ' ' + test_df['category']

    #Эмбеддинги (батчами) — размер батча 32 для T4
    embed_model = AutoModel.from_pretrained(MODEL_EMBED_PATH, torch_dtype=torch.float16, device_map="auto")
    embed_processor = AutoProcessor.from_pretrained(MODEL_EMBED_PATH)
    X_test = embed_texts_batch(test_df['text'].tolist(), embed_model, embed_processor, batch_size=32)
    del embed_model, embed_processor
    gc.collect()
    torch.cuda.empty_cache()

    #Предсказание вердиктов
    artifact = joblib.load(ARTIFACT_PATH)
    verdicts = predict_verdict(test_df, X_test, artifact)

    #Генерация объяснений (LLM) — размер батча 32
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_LLM_PATH)
        llm_model = AutoModelForCausalLM.from_pretrained(
            MODEL_LLM_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        results = generate_explanations_batch(
            test_df['text'].tolist(),
            test_df['category'].tolist(),
            verdicts,
            llm_tokenizer,
            llm_model,
            batch_size=32
        )
    except Exception as e:
        print(f"LLM error: {e}. Using rule-based fallback.")
        results = []
        for idx, row in test_df.iterrows():
            results.append(generate_answer_rule(row, verdicts[idx]))

    out_df = pd.DataFrame({'id': test_df['id'], 'result': results})
    out_df.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    main()