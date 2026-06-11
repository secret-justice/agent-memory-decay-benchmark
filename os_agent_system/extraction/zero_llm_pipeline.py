"""
Zero LLM Pipeline v3.1
======================
Enhanced classification with:
- Implicit preference patterns (every time, always, previously)
- Clause-level splitting for long sentences
- Better negation detection (disable, stop, never)
- Tool indicator tie-breaking
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LLMMode(Enum):
    FULL = "full"
    DEGRADED = "degraded"
    ZERO = "zero"


@dataclass
class ZeroLLMConfig:
    mode: LLMMode = LLMMode.DEGRADED
    max_summary_len: int = 200
    fact_min_len: int = 10
    enable_keyword_extraction: bool = True


class ZeroLLMPipeline:
    """
    Zero LLM Pipeline v3.1

    When LLM unavailable, use rules:
    1. Fact extraction: sentence split + pattern match
    2. Summary: truncation + key sentence extraction
    3. Classification: multi-dimensional scoring
    4. Conflict detection: semantic similarity rules
    """

    FACT_PATTERNS = [
        r"(?:\u6211|\u7528\u6237)(?:\u559c\u6b22|\u504f\u597d|\u4e60\u60ef|\u5e38\u7528|\u7231\u7528|\u503e\u5411)",
        r"(?:prefer|like|use|favorite|\u5e38\u7528)",
        r"(?:\u51b3\u5b9a|\u9009\u62e9|\u91c7\u7528|\u4f7f\u7528\u4e86?|\u6539\u7528)",
        r"(?:decided|chose|switched|using|adopted)",
        r"(?:\u662f|\u7b49\u4e8e|\u610f\u5473\u7740|\u8868\u793a|\u5b9a\u4e49\u4e3a)",
        r"(?:is|equals|means|defined as)",
        r"(?:\u5b8c\u6210\u4e86?|\u4fee\u590d\u4e86?|\u90e8\u7f72\u4e86?|\u521b\u5efa\u4e86?|\u5220\u9664\u4e86?)",
        r"(?:completed|fixed|deployed|created|deleted)",
        r"(?:\u9047\u5230|\u53d1\u73b0|\u51fa\u73b0\u4e86?|\u62a5\u9519|\u5931\u8d25)",
        r"(?:encountered|found|error|failed|issue)",
    ]

    CATEGORY_KEYWORDS = {
        "preference": ["\u559c\u6b22", "\u504f\u597d", "prefer", "like", "favorite",
                       "\u5e38\u7528", "\u7231\u7528", "\u503e\u5411", "always use",
                       "\u4e60\u60ef", "\u9ed8\u8ba4", "default", "\u5e0c\u671b", "\u60f3\u8981",
                       # colloquial preference keywords
                       "yyds", "\u771f\u9999", "\u7edd\u4e86", "\u9760\u8c31", "\u597d\u4f7f",
                       "\u4e2d\u7528", "\u9876", "\u725b", "\u8d5e", "\u63a8",
                       "\u522b\u7528", "\u522b\u518d", "\u53d7\u4e0d\u4e86", "\u518d\u4e5f\u4e0d",
                       "\u8d76\u7d27\u6dd8\u6c70", "\u6dd8\u6c70", "\u6362\u6389", "\u6362\u6210",
                       "\u7814\u9999", "\u771f\u9876", "\u771f\u725b", "\u7edd\u5b50",
                       "\u592a\u597d\u7528", "\u592a\u68d2\u4e86", "\u592a\u5f3a\u4e86",
                       "\u5c01\u795e", "\u8d62\u9ebb", "\u7701\u5fc3", "\u9760\u8c31",
                       "\u5c31\u5b83\u4e86", "\u5fc5\u987b", "\u7edf\u4e00", "\u5168\u90e8"],
        "knowledge": ["\u662f", "\u7b49\u4e8e", "\u5b9a\u4e49", "\u89c4\u5219", "is", "equals",
                      "rule", "running", "\u8fd0\u884c", "\u7aef\u53e3", "port", "package",
                      "\u5b89\u88c5", "\u7f16\u8f91", "edit", "\u652f\u6301", "supports",
                      "\u901a\u5e38", "\u7528\u6765", "\u7528\u4f5c", "\u5e38\u7528\u4f5c",
                      "\u6027\u80fd", "\u7aef\u53e3", "\u7248\u672c", "\u9ed8\u8ba4\u7aef\u53e3"],
        "episode": ["\u5b8c\u6210", "\u4fee\u590d", "\u90e8\u7f72", "\u505a\u4e86", "did",
                    "fixed", "completed", "\u5b89\u88c5", "install", "\u914d\u7f6e", "configure",
                    "issue", "bug", "timeout", "\u8c03\u8bd5", "debug"],
        "failure": ["\u9519\u8bef", "\u5931\u8d25", "bug", "error", "fail", "crash",
                    "\u62a5\u9519", "\u4e0d\u5de5\u4f5c"],
        "task": ["\u5e2e\u6211", "\u8bf7", "\u6267\u884c", "\u8fd0\u884c", "install",
                 "deploy", "\u914d\u7f6e", "setup", "check", "\u68c0\u67e5"],
        "trivial": ["hi", "hello", "ok", "\u597d\u7684", "\u55ef", "\u8c22\u8c22", "thanks"],
    }

    # Tool indicators for context-aware scoring
    TOOL_INDICATORS = frozenset({
        "vim", "nvim", "emacs", "vscode", "neovim", "python", "docker",
        "nginx", "redis", "mysql", "postgres", "postgresql", "git", "ssh",
        "json", "yaml", "xml", "csv", "toml", "pytest", "unittest",
        "httpx", "requests", "flask", "django", "fastapi", "react",
        "vue", "angular", "typescript", "rust", "golang", "go",
        "tmux", "zsh", "bash", "fish", "alacritty", "kitty",
        "pep8", "black", "prettier", "eslint", "pylint", "gdb", "lldb",
        "meilisearch", "elasticsearch", "kafka", "rabbitmq", "celery",
        "ubuntu", "centos", "debian", "alpine", "macos", "windows",
        "podman", "k3s", "helm", "terraform", "ansible", "jenkins",
        "github", "gitlab", "bitbucket", "circleci", "travis",
        "jupyter", "notebook", "colab", "pycharm", "intellij",
        "chrome", "firefox", "safari", "edge", "curl", "wget",
        "nodejs", "npm", "yarn", "pnpm", "bun", "deno",
        "postgres", "sqlite", "mongodb", "cassandra", "elasticsearch",
        "starship", "oh-my-zsh", "powerlevel10k",
    })

    def __init__(self, config: ZeroLLMConfig = None):
        self._config = config or ZeroLLMConfig()
        self._fact_patterns = [re.compile(p, re.IGNORECASE) for p in self.FACT_PATTERNS]

    def extract_facts(self, text: str) -> List[Dict[str, Any]]:
        sentences = self._split_sentences(text)
        facts = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < self._config.fact_min_len:
                continue
            matched = False
            for pattern in self._fact_patterns:
                if pattern.search(sent):
                    matched = True
                    break
            if matched:
                category = self._classify(sent)
                facts.append({
                    "content": sent,
                    "category": category,
                    "confidence": 0.6,
                    "method": "rule",
                })
        return facts

    def summarize(self, text: str, max_len: int = None) -> str:
        max_len = max_len or self._config.max_summary_len
        sentences = self._split_sentences(text)
        key_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            for pattern in self._fact_patterns:
                if pattern.search(sent):
                    key_sentences.append(sent)
                    break
            if sum(len(s) for s in key_sentences) >= max_len:
                break
        if not key_sentences:
            return text[:max_len]
        summary = " ".join(key_sentences)
        return summary[:max_len] if len(summary) > max_len else summary

    def extract_keywords(self, text: str) -> List[str]:
        if not self._config.enable_keyword_extraction:
            return []
        words = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', text)
        stop_words = {
            "\u7684", "\u4e86", "\u662f", "\u5728", "\u6211", "\u6709",
            "\u548c", "\u5c31", "\u4e0d", "\u4eba", "\u90fd", "\u4e00",
            "\u4e0a", "\u4e5f", "\u5f88", "\u5230", "\u8bf4", "\u8981",
            "the", "and", "for", "that", "this", "with", "are", "was",
        }
        return [w for w in words if w.lower() not in stop_words]

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r'[.!?;\n]|\u3002|\uff01|\uff1f|\uff1b|\uff0c', text)
        return [p.strip() for p in parts if p.strip()]

    def _split_clauses(self, text: str) -> List[str]:
        """Split long text into clauses for per-clause scoring."""
        # Split on: comma, semicolon, "and", "but", "then", period, etc.
        clause_seps = r'[,;\uff0c\uff1b]|\u4f46\u662f|\u7136\u540e|\u6700\u540e|\u53d1\u73b0|\u89c9\u5f97|but|then|,\s*so\s*'
        clauses = re.split(clause_seps, text)
        return [c.strip() for c in clauses if c.strip() and len(c.strip()) >= 4]

    def _classify(self, text: str) -> str:
        """Multi-dimensional scoring classification (v3.1)."""
        text_lower = text.lower().strip()

        if len(text_lower) <= 1:
            return "trivial"

        # For long sentences, try clause-level classification first
        if len(text) > 30:
            clauses = self._split_clauses(text)
            if len(clauses) >= 2:
                clause_scores = []
                for clause in clauses:
                    cat = self._score_single(clause)
                    clause_scores.append(cat)
                # If any clause is preference, the whole sentence is preference
                if "preference" in clause_scores:
                    return "preference"
                # Otherwise use the most common non-trivial category
                non_trivial = [c for c in clause_scores if c != "trivial"]
                if non_trivial:
                    from collections import Counter
                    return Counter(non_trivial).most_common(1)[0][0]

        return self._score_single(text)

    def classify(self, text: str) -> str:
        """Public API: classify a text into category."""
        return self._classify(text)

    def _score_single(self, text: str) -> str:
        """Score a single text unit (sentence or clause)."""
        text_lower = text.lower().strip()
        if len(text_lower) <= 1:
            return "trivial"

        scores = {cat: 0.0 for cat in self.CATEGORY_KEYWORDS}

        # === Dimension 1: Keyword matching (base score) ===
        for cat, keywords in self.CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[cat] += 1.0

        # === Dimension 2: Pattern matching (pattern-based boost) ===
        preference_patterns = [
            (r'(?:\u6211|\u7528\u6237)(?:\u559c\u6b22|\u504f\u597d|\u4e60\u60ef|\u5e38\u7528|\u7231\u7528|\u503e\u5411|\u5e0c\u671b|\u60f3\u8981)', 3.0),
            (r'(?:prefer|like|love|want|wish)', 3.0),
            (r'(?:\u4ee5\u540e|\u4e4b\u540e|\u4ece\u4eca\u4ee5\u540e|\u5f80\u540e)', 2.5),
            (r'(?:\u9ed8\u8ba4|default)\s*(?:\u7528|\u4f7f\u7528|to|use)', 2.5),
            (r'(?:always|\u4e00\u76f4|\u6bcf\u6b21|\u603b\u662f)\s*(?:\u7528|\u4f7f\u7528|use)', 3.0),
            (r'(?:\u522b|\u4e0d\u8981|\u4e0d\u7528|\u522b\u518d|\u505c\u6b62)\s*(?:\u7528|\u4f7f\u7528|\u7ed9\u6211)', 2.5),
            (r'(?:please)\s*(?:\u7528|\u4f7f\u7528|default|use)', 2.0),
            (r'(?:\u6539\u7528|\u6362\u6210|\u5207\u6362\u5230|switch\s+to|migrate\s+to)', 2.5),
            (r'(?:\u9075\u5faa|\u9075\u5b88|\u6309\u7167|follow)\s*\w+', 1.5),
            (r'(?:\u66f4\u9002\u5408|\u6700\u9002\u5408|\u6700\u597d|best|better)', 2.0),
            (r'(?:\u6211\u89c9\u5f97|\u6211\u8ba4\u4e3a|I\s+think|I\s+feel)', 1.5),
            # NEW: Implicit preference patterns
            (r'(?:\u6bcf\u6b21|\u4e00\u76f4|\u603b\u662f|\u4e00\u76f4\u90fd|\u6bcf\u6b21\u90fd)', 3.0),
            (r'(?:\u4e0a\u6b21|\u4e4b\u524d|\u4ee5\u524d)\s*(?:\u4e5f|\u90fd)?\s*(?:\u662f|\u7528)', 2.5),
            (r'(?:\u9879\u76ee\u91cc|\u56e2\u961f|\u751f\u4ea7\u73af\u5883)\s*(?:\u90fd|\u4e5f)?\s*(?:\u662f|\u7528)', 2.5),
            (r'(?:\u611f\u89c9|\u89c9\u5f97)\s*\w*\s*(?:\u6bd4|\u66f4|\u597d)', 2.0),
            (r'(?:\u51b3\u5b9a|decided)\s*(?:\u5168\u90e8|all|\u90fd)?\s*(?:\u8fc1\u79fb|\u5207\u6362|migrate|switch)', 2.5),
            # NEW: "disable X" = anti-preference
            (r'(?:disable|\u7981\u7528|\u5173\u95ed|turn\s+off|\u505c\u7528)', 2.5),
            # NEW: Indirect preference indicators
            (r'(?:\u63a8\u8350|\u5b89\u5229|\u79cd\u8349|\u63a8\u4e86)', 2.5),
            (r'(?:\u8981\u6c42|\u547d\u4ee4|\u7edf\u4e00\u7528|\u7edf\u4e00\u4f7f\u7528)', 3.0),
            (r'(?:\u8fd8\u662f\u7528|\u8fd8\u662f\u6362\u6210|\u6539\u4e3a\u7528)', 2.5),
            # NEW: Temporal preference transition
            (r'(?:\u4ee5\u524d\u4e60\u60ef|\u4ee5\u524d\u7528|\u66fe\u7ecf\u7528|\u4e4b\u524d\u7528)', 2.0),
            (r'(?:\u73b0\u5728\u6539\u4e3a|\u73b0\u5728\u7528|\u5168\u9762\u8f6c|\u5f00\u59cb\u7528)', 3.0),
            # NEW: "觉得X比Y好" comparison preference
            (r'(?:\u89c9\u5f97|feel)\s*\w+\s*(?:\u6bd4|\u66f4)\s*\w+\s*(?:\u597d|\u9002\u5408|\u5f3a)', 2.5),
            # 口语化偏好：行吧/算了/得了/哎/就这样吧/将就着/凑合
            (r'(?:行吧|算了|哎|得了|就这样吧|将就着|凑合)', 2.5),
            # "还是X吧/好了/得了"
            (r'(?:还是|就)\s*\w+(?:吧|好了|得了)', 3.0),
            # "说实话/说真的/其实 + 还是/觉得"
            (r'(?:说实话|说真的|其实)\s*(?:还是|觉得)', 2.5),
            # "听你的用X"
            (r'听你的', 2.5),
            # "习惯了吧/习惯了用X"
            (r'习惯(?:了)?吧', 2.5),
            # "不折腾了 + 用X"
            (r'不折腾了', 2.0),
            # "无所谓了 + 用X"
            (r'无所谓了', 2.0),
            # "要不试试X"
            (r'要不试试', 2.0),
            # "就这个吧"
            (r'就这个吧', 2.5),
            # "那就X吧"
            (r'那就\w+吧', 3.0),
            # 隐晦偏好
            (r'(?:老规矩|惯例|你知道|你懂的|跟上次一样|按惯例|说好的|老样子)', 3.5),
            (r'每次.{0,10}(?:都是|都用|都这样)', 2.5),
            # 强度梯度
            (r'(?:稍微|比较|强烈|非\w+不可|只能|必须|唯一)', 2.5),
            (r'(?:倾向于|喜欢|推荐|选择)', 2.0),
            # FN fix: 比较偏好 (require personal context)
            (r'(?:我|用户|觉得|感觉|我们).{0,8}比\w+(?:好用|好|强|快|适合|方便)', 2.5),
            # FN fix: "反正都是"
            (r'反正都是', 2.5),
            # FN fix: "也还行/也不错"
            (r'也(?:还行|不错|挺好|可以)', 2.0),
            # FN fix: "建议用/推荐用"
            (r'(?:建议|推荐)\s*(?:用|使用)', 2.5),
            # FN fix: "反对/拒绝/排除/不允许" + tool
            (r'(?:反对|拒绝|排除|不允许|禁止)\s*(?:用|使用)', 3.0),
            # FN fix: "以后都用/以后用"
            (r'以后(?:都)?(?:用|使用)', 3.0),
            # FN fix: 短句偏好 "用X"
            (r'^用[\u4e00-\u9fff\w]{5,}$', 1.5),
        ]
        # === Conditional preference patterns (Exp77 fix) ===
        # Only fire when personal context present
        conditional_pref_pats = [
            # 我X用A Y用B - personal conditional
            (r'(?:我|我们)\s*[一-鿿\w]*用[一-鿿\w]+[一-鿿\w]*用[一-鿿\w]+', 3.0),
            # 建议/推荐用X
            (r'(?:建议|推荐|最好)\s*[一-鿿\w]*用[一-鿿\w]+', 2.5),
            # 全部用X / 统一用X
            (r'(?:全部|统一)\s*(?:用|切换到|迁移到)[一-鿿\w]+', 3.0),
            # 强烈推荐
            (r'(?:强烈|强力|剧烈)\s*(?:推荐|建议)', 3.0),
            # 不是说X不好只是Y更合适
            (r'(?:不是说|不是)[一-鿿\w]*(?:不好|不行|不能)', 2.5),
        ]
        for pat, boost in conditional_pref_pats:
            if re.search(pat, text_lower):
                scores['preference'] += boost

        colloquial_pref_pats = [
            # Ultra-colloquial
            (r'(?:yyds|\u771f\u9999|\u7edd\u4e86|\u771f\u9876|\u771f\u725b|\u7edd\u5b50)', 4.0),
            (r'(?:\u9760\u8c31|\u597d\u4f7f|\u4e2d\u7528)', 3.0),
            (r'(?:\u592a\u597d\u7528\u4e86|\u592a\u68d2\u4e86|\u592a\u5f3a\u4e86|\u592a\u65b9\u4fbf\u4e86)', 3.5),
            (r'(?:\u592a\u6298\u817e\u4e86|\u592a\u91cd\u4e86|\u592a\u6162\u4e86|\u592a\u5783\u573e\u4e86)', 3.0),
            (r'(?:\u8c01\u8fd8\u7528|\u73b0\u5728\u8c01\u4e0d|\u8c01\u8fd8\u4e0d)', 3.5),
            (r'(?:\u522b\u7528\u4e86|\u522b\u518d\u7528|\u8d76\u7d27\u6dd8\u6c70|\u6dd8\u6c70\u5427)', 3.0),
            (r'(?:\u53d7\u4e0d\u4e86|\u518d\u4e5f\u4e0d\u7528|\u518d\u4e5f\u4e0d\u60f3)', 3.5),
            (r'(?:\u6bd4\u597d\u7528|\u6bd4\u9760\u8c31|\u6bd4\u5f3a|\u6bd4\u5feb)', 2.5),
            (r'(?:\u4ee5\u540e\u5c31\u5b83\u4e86|\u4ee5\u540e\u5c31\u7528\u8fd9\u4e2a)', 3.5),
            (r'(?:\u7ed9\u6211\u6574\u4e00\u4e2a|\u7ed9\u6211\u6765\u4e00\u4e2a)', 2.5),
            (r'(?:\u8fd8\u662f\u8fd9\u4e2a\u9760\u8c31|\u8fd8\u662f\u8fd9\u4e2a\u597d)', 3.5),
            (r'(?:\u8fd9\u73a9\u610f\u513f|\u8fd9\u4e1c\u897f)\s*(?:\u771f|\u592a|\u5f88)', 3.0),
            (r'(?:\u6211\u4e00\u822c\u90fd|\u6211\u901a\u5e38\u90fd|\u6211\u4e60\u60ef)', 3.0),
            (r'(?:\u54a8\u8be2\u70b9\u540d\u8981\u7528|\u5ba2\u6237\u8981\u6c42|\u5ba2\u6237\u70b9\u540d)', 3.5),
            # Ultra-colloquial extended
            (r'(?:\u5c01\u795e|\u8d62\u9ebb\u4e86|\u76f4\u63a5\u5c01\u795e|\u7b80\u76f4\u662f)', 4.0),
            (r'(?:\u6709\u4e00\u8bf4\u4e00|\u5c31\u662f\u8bf4|\u4e00\u628a\u68ad)', 3.0),
            (r'(?:\u5feb\u5230\u98de\u8d77|\u7701\u5fc3\u591a\u4e86|\u6210\u719f\u7a33\u5b9a)', 3.0),
            (r'(?:\u5c31\u5b83\u4e86|\u9009\u8fd9\u4e2a|\u6211\u9009)[\u4e00-\u9fff\w]+', 3.0),
            (r'[\u4e00-\u9fff\w]+\u6bd4[\u4e00-\u9fff\w]+(?:\u597d|\u5f3a|\u5feb|\u7a33|\u7701\u5fc3|\u9760\u8c31)', 2.5),
            (r'(?:\u5fc5\u987b|\u7edf\u4e00|\u5168\u90e8)\s*(?:\u7528|\u9009|\u8fc1\u79fb|\u5207\u6362)?[\u4e00-\u9fff\w]+', 3.0),
            (r'[\u4e00-\u9fff\w]+(?:\u5c01\u795e|\u8d62\u9ebb|\u7edd\u4e86|\u9876)', 3.5),
            # Behavioral implicit
            (r'(?:\u6211\s*)?(?:\u6bcf\u4e2a|\u6bcf\u6b21|\u6240\u6709)\s*[\u4e00-\u9fff\w]+\s*(?:\u90fd|\u5c31)\s*(?:\u7528|\u52a0|\u5f00|\u8bbe\u7f6e|\u914d\u7f6e)', 3.0),
            # Traditional Chinese
            (r'(?:\u4ee5\u5f8c|\u9084\u662f|\u89ba\u5f97)\s*[\u4e00-\u9fff\w]*(?:\u7528|\u5beb|\u5199)', 2.5),
        ]
        for pat, boost in colloquial_pref_pats:
            if re.search(pat, text_lower):
                scores["preference"] += boost

        knowledge_patterns = [
            (r'(?:\u662f|\u7b49\u4e8e|means|equals|defined\s+as)\s', 2.0),
            (r'(?:\u9ed8\u8ba4|default)\s*(?:\u7aef\u53e3|port|\u503c|value)\s*(?:\u662f|is|=|:)', 3.0),
            (r'(?:\u7aef\u53e3|port)\s*\d+', 2.5),
            (r'(?:\u8fd0\u884c|running|\u76d1\u542c|listen)\s+(?:\u5728|on|at)\s*\d+', 2.5),
            (r'(?:\u5b89\u88c5|install)\s+\w+\s*(?:\u9700\u8981|requires)', 2.0),
            (r'(?:\u652f\u6301|supports|provides?)\s+\w+\s+(?:\u529f\u80fd|feature|type)', 2.0),
        ]
        episode_patterns = [
            (r'(?:\u5b8c\u6210\u4e86?|\u4fee\u590d\u4e86?|\u90e8\u7f72\u4e86?|\u521b\u5efa\u4e86?|\u5220\u9664\u4e86?|\u89e3\u51b3\u4e86?)', 2.5),
            (r'(?:completed|fixed|deployed|created|deleted|resolved|solved)', 2.5),
            (r'(?:\u505a\u4e86|\u641e\u4e86|\u5f04\u4e86|\u641e\u5b9a\u4e86)', 2.0),
            (r'(?:\u9047\u5230\u4e86?|\u53d1\u73b0\u4e86?|\u51fa\u73b0\u4e86?|\u62a5\u9519|\u5931\u8d25)', 2.0),
            (r'(?:\u6628\u5929|\u4eca\u5929|\u4e0a\u5468|last\s+(?:week|time))', 1.5),
            (r'(?:\u8c03\u8bd5\u4e86|\u8bd5\u4e86|\u7528\u4e86|\u8dd1\u4e86)', 1.5),
            # Fix Exp87: temporal episode patterns
            (r'(?:\u524d\u5929|\u524d\u51e0\u5929|\u6628\u665a|\u4e0a\u4e2a\u6708|\u53bb\u5e74)', 2.0),
            (r'(?:\u5347\u7ea7\u5230|\u5347\u7ea7\u4e3a|\u5347\u7ea7\u4e86|\u66f4\u65b0\u5230|\u66f4\u65b0\u4e86)', 2.5),
            (r'(?:\u8e29\u4e86|\u8e29\u5751|\u5751\u4e86|\u5403\u4e86.*\u4e8f|\u8e29\u8fc7)', 3.0),
            (r'(?:\u82b1\u4e86|\u8017\u65f6|\u8017\u8d39|\u5f04\u4e86\u5f88\u4e45)', 2.0),
        ]

        for pat, boost in preference_patterns:
            if re.search(pat, text_lower):
                scores["preference"] += boost
        for pat, boost in knowledge_patterns:
            if re.search(pat, text_lower):
                scores["knowledge"] += boost
        for pat, boost in episode_patterns:
            if re.search(pat, text_lower):
                scores["episode"] += boost

        # === Dimension 3: Context signals ===
        # "用X" pattern with tool indicator
        tool_matches = re.findall(r'(?:\u7528|\u4f7f\u7528|use|using)\s+(\w+)', text_lower)
        for tool in tool_matches:
            if tool in self.TOOL_INDICATORS:
                scores["preference"] += 2.5

        # "是X型/类" pattern indicates knowledge
        if re.search(r'\u662f\w+(?:\u578b|\u7c7b|\u5f0f)\u7684?', text_lower):
            scores["knowledge"] += 2.0

        # Negation with tool = anti-preference
        neg_tool = re.search(r'(?:\u4e0d\u8981|\u522b|\u4e0d\u7528|\u7981\u6b62|never|dont|stop|disable)\s*(?:\u7528|\u4f7f\u7528|use)?\s*(\w+)', text_lower)
        if neg_tool:
            scores["preference"] += 3.0

        # NEW: "from X to Y" / "换成" pattern = preference for Y
        if re.search(r'(?:\u4ece|from)\s*\w+\s*(?:\u6362\u6210|\u6539\u5230|\u8fc1\u79fb\u5230|switched?\s+to|migrat\w+\s+to)', text_lower):
            scores["preference"] += 2.5

        # NEW: "决定全部迁移" = strong preference
        if re.search(r'(?:\u51b3\u5b9a|decided)\s*(?:\u5168\u90e8|all)?\s*(?:\u8fc1\u79fb|\u5207\u6362|migrate|switch)', text_lower):
            scores["preference"] += 2.0

        # NEW: "最合适/最适合" in context of comparison = preference
        if re.search(r'(?:\u6700\u9002\u5408|\u6700\u9002\u5408|\u6700\u597d\u7528)', text_lower):
            scores["preference"] += 2.0

        # === Decision ===
        #         # Conditional pattern boost (before max_score check)
        # "小项目用SQLite大项目用PostgreSQL" = preference
        _early_cond = re.search(
            r'(?:小项目|大项目|测试环境|生产环境|本地|线上|前端|后端|轻量级|重量级|静态|动态)\s*(?:用|上用)\s*[a-zA-Z0-9_\-一-鿿]+',
            text_lower,
        )
        if _early_cond:
            # Check for 2+ tool assignments
            _early_tools = re.findall(r'(?:用|上用)\s*([a-zA-Z][a-zA-Z0-9_\-]*)', text_lower)
            _early_in = [t for t in _early_tools if t in self.TOOL_INDICATORS]
            if len(_early_in) >= 2:
                scores["preference"] += 5.0

        # Pre-score boost: 3+ multi-subject tool-using patterns
        # Require 3+ tools to avoid FP on factual comparisons like "前端项目用React，后端项目用Go"
        _pre_subj_tool = re.findall(r'[\u4e00-\u9fff]{2,6}\s*(?:\u7528|\u4e0a\u7528)\s*([a-zA-Z][a-zA-Z0-9_\-]*)', text_lower)
        if len(_pre_subj_tool) >= 3:
            scores["preference"] += 3.0

        max_score = max(scores.values())
        # Single short word: trivial for greetings, knowledge for tech terms
        if len(text_lower.strip()) <= 4 and max_score <= 1.0:
            _trivial_words = {'hi', 'hello', 'ok', '好的', '嗯', '谢谢', 'thanks', 'yes', 'no'}
            if text_lower.strip() in _trivial_words:
                return 'trivial'
            # Fix: short tool preference
            _short_tool_match = re.match(r'(?:用|使用)[a-zA-Z]+', text_lower.strip())
            if _short_tool_match:
                _st = re.findall(r'[a-zA-Z]+', text_lower.strip())
                if any(t in self.TOOL_INDICATORS for t in _st):
                    return 'preference'
            return 'knowledge'

        if max_score <= 0:
            return "knowledge"

        pref_score = scores["preference"]
        know_score = scores["knowledge"]
        epi_score = scores["episode"]

        # FP fix: "建议/推荐" in knowledge context is lower preference
        if re.search(r'(?:支持|supports|provides?|用于|用来|通常|标准|默认)\s', text_lower):
            know_score += 2.0
        
        # NEW: Anti-pattern — "X用来/用作/常用作 Y" is knowledge, not preference
        # Very strong pattern: "常用作/用来/用作" always indicates knowledge
        _usage_as_pattern = re.search(r'(?:常用作|用来|用作)\s*[一-鿿\w]+', text_lower)
        if _usage_as_pattern:
            know_score += 5.0  # Very strong knowledge signal
        # "用于" is also knowledge when not preceded by preference words
        if '用于' in text_lower:
            _pref_before = re.search(r'(?:我|用户)(?:喜欫|偏好|习惯).{0,10}用于', text_lower)
            if not _pref_before:
                know_score += 3.0

        # NEW: Conditional preference detection
        # "X场景用A，Y场景用B" pattern
        if re.search(r'(?:项目|场景|环境|小项目|大项目|测试环境|生产环境)\s*用', text_lower):
            pref_score += 2.0
        if re.search(r'用\w+\s*[，,]\s*\w+用', text_lower):
            pref_score += 1.5

        # 条件偏好检测增强：X用A Y用B 模式
        _cond_pattern1 = re.search(
            r'(?:小项目|大项目|测试环境|生产环境|本地|线上|前端|后端|轻量级|重量级|mac|linux|windows|静态|动态|脚本|服务|测试|开发)\s*(?:用|上用|资源用|请求用)\s*\w+',
            text_lower,
        )
        _cond_pattern2 = re.search(
            r'(?:用|上用|资源用|请求用)\s*\w+\s*(?:大项目|小项目|生产|测试|线上|本地|后端|前端|重量级|轻量级|linux|windows|mac|动态|静态)\s*(?:用|上用|资源用|请求用)',
            text_lower,
        )
        # General: any two "X用A Y用B" patterns with different subjects
        _cond_pattern3 = re.search(
            r'[\u4e00-\u9fff\w]{1,6}(?:用|上用)\s*[\w\-]+\s*[\u4e00-\u9fff\w]{1,6}(?:用|上用)',
            text_lower,
        )
        # Fix Exp87: compound "X用A Y用B Z用C" with 3+ segments
        _cond_pattern4 = re.search(
            r'[\u4e00-\u9fff\w]{1,6}(?:用|上用)\s*[\w\-]+\s*[\u4e00-\u9fff\w]{1,6}(?:用|上用)\s*[\w\-]+',
            text_lower,
        )
        if _cond_pattern1 or _cond_pattern2 or _cond_pattern3 or _cond_pattern4:
            pref_score += 3.0

        # 多工具并列模式：X用A Y用B Z用C
        _tool_listing = re.findall(r'(?:用|上用)\s*([a-zA-Z][a-zA-Z0-9_\-]*)', text_lower)
        if len(_tool_listing) >= 2:
            all_tools = [t for t in _tool_listing if t in self.TOOL_INDICATORS]
            # Fix Exp87: prefix match for truncated tools (pyte->pytest, go->go)
            if len(all_tools) < 2:
                for t in _tool_listing:
                    if t not in all_tools:
                        for indicator in self.TOOL_INDICATORS:
                            if indicator.startswith(t) and len(t) >= 3:
                                all_tools.append(t)
                                break
            if len(all_tools) >= 2:
                pref_score += 3.0

        # 否定+原因 = 强偏好信号
        if re.search(r'(?:不喜欢|不想用|不好用|太臃肿|太重|太慢|太复杂)\s*(?:了|啦)', text_lower):
            pref_score += 2.5

        # 主观词 = preference
        if re.search(r'(?:觉得|感觉|主观|个人|习惯性|本能)', text_lower):
            pref_score += 2.0

        # Anti-FP: strong knowledge context suppresses preference
        _strong_knowledge = re.search(r'(?:通常|用来|用作|标准|规范|支持|提供|运行在|监听在|端口是|版本是|默认端口|性能)', text_lower)
        if _strong_knowledge and pref_score < know_score * 1.5:
            know_score += 3.0

        # Anti-FP: "X通常用来Y" = knowledge not preference
        _usage_pattern = re.search(r'(?:通常|常用|一般)\s*(?:用来|用作|用于)', text_lower)
        if _usage_pattern:
            know_score += 5.0

        # Anti-FP: "X性能比Y强" = factual comparison, not preference
        _factual_compare = re.search(r'(?:性能|速度|效率|内存|资源)\s*比', text_lower)
        if _factual_compare:
            know_score += 10.0  # Very strong knowledge signal

        # Preference wins if it is at least 60% of the highest score

        # Use already-computed anti-FP signals
        if _factual_compare:
            know_score = max(know_score, pref_score + 5.0)
            scores["knowledge"] = max(scores["knowledge"], scores.get("preference", 0) + 5.0)
        if _usage_pattern:
            know_score = max(know_score, pref_score + 5.0)
            scores["knowledge"] = max(scores["knowledge"], scores.get("preference", 0) + 5.0)


            import sys as _dbg

        if pref_score > 0 and pref_score >= know_score * 0.6 and pref_score >= epi_score * 0.6:
            return "preference"

        return max(scores, key=scores.get)

    @property
    def mode(self) -> LLMMode:
        return self._config.mode

    @property
    def is_zero_llm(self) -> bool:
        return self._config.mode == LLMMode.ZERO

    def stats(self) -> Dict[str, Any]:
        return {
            "mode": self._config.mode.value,
            "fact_patterns": len(self.FACT_PATTERNS),
            "categories": list(self.CATEGORY_KEYWORDS.keys()),
        }
