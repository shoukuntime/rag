import re

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

from env_settings import EnvSettings


def chinese_to_int(s):
    """
    將中文數字（一至九十九）轉換為整數。
    """
    num_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}

    if '十' in s:
        parts = s.split('十')
        if parts[0] == '':
            total = 10
        else:
            total = num_map.get(parts[0], 0) * 10

        if len(parts) > 1 and parts[1] != '':
            total += num_map.get(parts[1], 0)
        return total
    else:
        return num_map.get(s, 0)


def sort_key_for_articles(article_id):
    """
    自訂排序鍵，用於處理 '17' 和 '17-1' 這樣的字串。
    """
    if '-' in article_id:
        parts = article_id.split('-')
        return (int(parts[0]), int(parts[1]))
    else:
        return (int(article_id), 0)


def parse_labor_law_with_chapters(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    chinese_num_map = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7,
        '八': 8, '九': 9, '十': 10, '十一': 11, '十二': 12
    }

    content = re.sub(r'法規名稱：.*\n', '', content)
    content = re.sub(r'修正日期：.*\n', '', content)

    result = []

    chapter_pattern = re.compile(r'\s*第\s*(\S+)\s*章[^\n]*\n([\s\S]*?)(?=\s*第 \S+ 章|\Z)')
    article_pattern = re.compile(r'第\s*(\d+(?:-\d+)?)\s*條\n([\s\S]*?)(?=\n\s*第|\Z)')

    # 更新後的正則表達式，可以捕獲 "第...條" 和 "第...條之一"
    ref_article_pattern = re.compile(r'第([一二三四五六七八九十]+)條(之一)?')

    for chapter_match in chapter_pattern.finditer(content):
        chinese_numeral, chapter_content = chapter_match.groups()
        chapter_number = chinese_num_map.get(chinese_numeral, 0)

        article_matches = article_pattern.findall(chapter_content)

        for article_number, article_text in article_matches:
            cleaned_content = article_text.strip()

            referenced_articles = set()
            for ref_match in ref_article_pattern.finditer(cleaned_content):
                cn_num, sub_part = ref_match.groups()
                num = chinese_to_int(cn_num)
                if num > 0:
                    article_ref_str = str(num)
                    # 如果匹配到 "之一"，則在編號後加上 "-1"
                    if sub_part:
                        article_ref_str += "-1"
                    referenced_articles.add(article_ref_str)

            result.append([
                str(chapter_number),
                article_number,
                cleaned_content,
                # 使用新的排序鍵進行排序
                sorted(list(referenced_articles), key=sort_key_for_articles)
            ])

    return result


env_settings = EnvSettings()

embeddings = GoogleGenerativeAIEmbeddings(model=env_settings.EMBEDDING_MODEL, google_api_key=env_settings.GOOGLE_API_KEY)


def ingest_data():
    file_path = '../data/labor_law.txt'
    parsed_law = parse_labor_law_with_chapters(file_path)
    print(f"Parsed {len(parsed_law)} articles.")

    docs = []
    for chapter, article_num, content, refs in parsed_law:
        metadata = {
            "chapter": chapter,
            "article": article_num,
            "references": ", ".join(refs)
        }
        doc = Document(page_content=content, metadata=metadata)
        docs.append(doc)

    print(f"Created {len(docs)} documents.")

    store = PGVector(
        collection_name=env_settings.COLLECTION_NAME,
        connection=env_settings.POSTGRES_URI,
        embeddings=embeddings,
    )

    store.add_documents(docs)
    print("儲存完成!")


if __name__ == "__main__":
    ingest_data()
