# encoding: utf-8
# 指定文件编码为UTF-8，确保中文字符正确处理
from __future__ import absolute_import, division, print_function, unicode_literals  # Python 2/3 兼容性导入，确保代码在Python 2和3中都能运行

import json  # 导入Python标准库模块：
import os    # json - JSON数据处理
import re    # os - 操作系统接口
import shutil       # re - 正则表达式
import threading    # shutil - 文件操作工具
import warnings     # threading - 多线程支持
                    # warnings - 警告处理
import six          # six库：Python 2/3兼容性工具
from django.conf import settings                                    # settings - Django项目配置
from django.core.exceptions import ImproperlyConfigured             # ImproperlyConfigured - 配置错误异常
from datetime import datetime                                       # datetime - 日期时间处理
from django.utils.encoding import force_str                          # force_str - 强制转换为字符串
from haystack.backends import BaseEngine, BaseSearchBackend, BaseSearchQuery, EmptyResults, log_query    # Haystack搜索框架相关导入
from haystack.constants import DJANGO_CT, DJANGO_ID, ID
from haystack.exceptions import MissingDependency, SearchBackendError, SkipDocument
from haystack.inputs import Clean, Exact, PythonData, Raw
from haystack.models import SearchResult
from haystack.utils import get_identifier, get_model_ct
from haystack.utils import log as logging
from haystack.utils.app_loading import haystack_get_model
from jieba.analyse import ChineseAnalyzer   # 结巴中文分词器
from whoosh import index                     # Whoosh搜索引擎主模块
from whoosh.analysis import StemmingAnalyzer  # 词干分析器
from whoosh.fields import BOOLEAN, DATETIME, IDLIST, KEYWORD, NGRAM, NGRAMWORDS, NUMERIC, Schema, TEXT   
from whoosh.fields import ID as WHOOSH_ID                      # 重命名避免冲突
from whoosh.filedb.filestore import FileStorage, RamStorage    # 文件存储和内存存储
from whoosh.highlight import ContextFragmenter, HtmlFormatter  # 搜索结果高亮
from whoosh.highlight import highlight as whoosh_highlight     # 重命名避免冲突
from whoosh.qparser import QueryParser      # 查询解析器
from whoosh.searching import ResultsPage    # 分页结果
from whoosh.writing import AsyncWriter      # 异步写入器

try:
    import whoosh   # 尝试导入whoosh搜索引擎库
except ImportError:   # 如果导入失败，说明系统未安装whoosh库
    raise MissingDependency(
        "The 'whoosh' backend requires the installation of 'Whoosh'. Please refer to the documentation.")

# Handle minimum requirement.
if not hasattr(whoosh, '__version__') or whoosh.__version__ < (2, 5, 0):  # 检查最低版本要求
    raise MissingDependency(
        "The 'whoosh' backend requires version 2.5.0 or greater.")  # 注释："Bubble up the correct error."，意思是让正确的错误信息向上传递，不要被掩盖

# Bubble up the correct error.

DATETIME_REGEX = re.compile(
    '^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})T(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})(\.\d{3,6}Z?)?$')
LOCALS = threading.local()  # 定义日期时间格式的正则表达式
LOCALS.RAM_STORE = None     # 用于匹配ISO 8601格式的日期时间字符串，如："2024-01-15T14:30:00.123456Z"


class WhooshHtmlFormatter(HtmlFormatter):
    """
    This is a HtmlFormatter simpler than the whoosh.HtmlFormatter.
    We use it to have consistent results across backends. Specifically,
    Solr, Xapian and Elasticsearch are using this formatting.
    """
    template = '<%(tag)s>%(t)s</%(tag)s>'
    # 高亮模板说明：
    # %(tag)s - 会被替换为高亮标签（如<em>、<span>等）
    # %(t)s   - 会被替换为需要高亮的文本内容
    # 示例：搜索"Python"时，可能生成：<em>Python</em>

class WhooshSearchBackend(BaseSearchBackend):
    # Word reserved by Whoosh for special use.
    # Whoosh保留的关键字列表，这些词在搜索查询中有特殊含义
    # 在构建搜索查询时，这些词会被特殊处理（如转义）
    RESERVED_WORDS = (
        'AND',
        'NOT',
        'OR',
        'TO',
    )

    # Characters reserved by Whoosh for special use.
    # The '\\' must come first, so as not to overwrite the other slash
    # replacements.
    RESERVED_CHARACTERS = (
        '\\', '+', '-', '&&', '||', '!', '(', ')', '{', '}',
        '[', ']', '^', '"', '~', '*', '?', ':', '.',
    )

    def __init__(self, connection_alias, **connection_options):
    # 调用父类BaseSearchBackend的初始化方法
    # 确保基本的搜索引擎后端功能正确初始化
        super(   # 标记设置是否完成，初始为False，在setup()方法中完成设置后变为True
            WhooshSearchBackend,
            self).__init__(
            connection_alias,
            **connection_options)
        self.setup_complete = False  # 是否使用文件存储，默认为True（使用磁盘文件存储索引）
    # 如果设置为False，则使用内存存储（RAM），适用于测试或临时搜索
        self.use_file_storage = True
    # 设置索引文档的大小限制，默认128MB
    # 防止单个文档过大影响搜索性能
        self.post_limit = getattr(
            connection_options,
            'POST_LIMIT',
            128 * 1024 * 1024)
        # 计算过程：128 * 1024KB * 1024B = 134,217,728字节
    
    # 获取索引文件存储路径
    # 这是索引数据在磁盘上的存储位置
        self.path = connection_options.get('PATH')
# 检查存储类型配置，如果明确设置为非'file'，则使用内存存储
        if connection_options.get('STORAGE', 'file') != 'file':
            self.use_file_storage = False  # 切换到内存存储模式
# 验证配置：如果使用文件存储但没有指定路径，抛出配置错误
        if self.use_file_storage and not self.path:
            raise ImproperlyConfigured(
                "You must specify a 'PATH' in your settings for connection '%s'." %
                connection_alias)  # 错误信息示例：必须为连接'default'在设置中指定'PATH'
# 初始化日志记录器，用于记录搜索操作和错误信息
    # 使用'haystack'命名空间的日志记录器
        
        self.log = logging.getLogger('haystack')

    def setup(self):
        """
        Defers loading until needed.
        """
        from haystack import connections
        new_index = False # 标记是否是新创建的索引

        # Make sure the index is there.确保索引目录存在（仅文件存储模式）
        if self.use_file_storage and not os.path.exists(self.path):
            os.makedirs(self.path)  # 创建索引存储目录
            new_index = True        # 标记为新索引，因为目录原本不存在

        if self.use_file_storage and not os.access(self.path, os.W_OK):   # 检查索引目录的写入权限（仅文件存储模式）
            raise IOError(
                "The path to your Whoosh index '%s' is not writable for the current user/group." %
                self.path)   # 错误信息：索引路径对当前用户/组不可写

        if self.use_file_storage:  # 根据存储模式初始化存储对象，文件存储：使用磁盘文件持久化存储索引
            self.storage = FileStorage(self.path)   
        else:  # 内存存储：使用RAM存储，速度快但重启后数据丢失
            global LOCALS  # 引用全局的线程本地存储

            if getattr(LOCALS, 'RAM_STORE', None) is None:
                LOCALS.RAM_STORE = RamStorage()

            self.storage = LOCALS.RAM_STORE# 使用线程本地的RAM存储

        self.content_field_name, self.schema = self.build_schema(
            connections[self.connection_alias].get_unified_index().all_searchfields()) # 构建Whoosh的schema（字段定义）并获取主内容字段名
        self.parser = QueryParser(self.content_field_name, schema=self.schema)         # 创建查询解析器，用于解析用户搜索字符串

        if new_index is True:  # 如果是新索引（目录刚创建），则创建全新的索引
            self.index = self.storage.create_index(self.schema)
        else:
            try:  # 尝试打开已存在的索引
                self.index = self.storage.open_index(schema=self.schema)  # 如果索引文件存在但为空，则重新创建索引
            except index.EmptyIndexError:
                self.index = self.storage.create_index(self.schema)

        self.setup_complete = True  # 标记设置完成，后续操作可以直接使用

    def build_schema(self, fields):  # 初始化基础字段schema，这些是Haystack必需的硬编码字段
        schema_fields = {
            ID: WHOOSH_ID(stored=True, unique=True), # 文档唯一标识符
            DJANGO_CT: WHOOSH_ID(stored=True),       # Django ContentType标识
            DJANGO_ID: WHOOSH_ID(stored=True),       # Django对象ID
        }
        # Grab the number of keys that are hard-coded into Haystack.记录初始键数量，用于后续验证是否成功添加了自定义字段
        # We'll use this to (possibly) fail slightly more gracefully later.主内容字段名，用于全文搜索
        initial_key_count = len(schema_fields)
        content_field_name = ''

        for field_name, field_class in fields.items():  # 遍历所有搜索字段，根据字段类型创建对应的Whoosh字段定义
            if field_class.is_multivalued:              # 处理多值字段（如标签列表、分类列表等）
                if field_class.indexed is False:
                    schema_fields[field_class.index_fieldname] = IDLIST(  # 不索引的多值字段：存储但不建立倒排索引
                        stored=True, field_boost=field_class.boost)
                else:  # 可索引的多值字段：支持评分和逗号分隔
                    schema_fields[field_class.index_fieldname] = KEYWORD(
                        stored=True, commas=True, scorable=True, field_boost=field_class.boost)
            elif field_class.field_type in ['date', 'datetime']:  # 日期时间字段， 可排序
                schema_fields[field_class.index_fieldname] = DATETIME(
                    stored=field_class.stored, sortable=True)
            elif field_class.field_type == 'integer':  # 整数字段
                schema_fields[field_class.index_fieldname] = NUMERIC(
                    stored=field_class.stored, numtype=int, field_boost=field_class.boost)
            elif field_class.field_type == 'float':    # 浮点数字段
                schema_fields[field_class.index_fieldname] = NUMERIC(
                    stored=field_class.stored, numtype=float, field_boost=field_class.boost)
            elif field_class.field_type == 'boolean':   # 布尔字段
                # Field boost isn't supported on BOOLEAN as of 1.8.2.
                schema_fields[field_class.index_fieldname] = BOOLEAN(
                    stored=field_class.stored)          # 注意：Whoosh 1.8.2版本不支持布尔字段的权重提升
            elif field_class.field_type == 'ngram':     # N-gram字段（用于模糊匹配）
                schema_fields[field_class.index_fieldname] = NGRAM(
                    minsize=3, maxsize=15, stored=field_class.stored, field_boost=field_class.boost)
            elif field_class.field_type == 'edge_ngram':  # 边缘N-gram字段（用于前缀搜索）
                schema_fields[field_class.index_fieldname] = NGRAMWORDS(minsize=2, maxsize=15, at='start',  # 从词首开始
                                                                        stored=field_class.stored,
                                                                        field_boost=field_class.boost)
            else:   # 默认情况：文本字段（使用中文分词器），关键配置：使用ChineseAnalyzer进行中文分词，替代默认的StemmingAnalyzer
                # schema_fields[field_class.index_fieldname] = TEXT(stored=True, analyzer=StemmingAnalyzer(), field_boost=field_class.boost, sortable=True)
                schema_fields[field_class.index_fieldname] = TEXT(
                    stored=True, analyzer=ChineseAnalyzer(), field_boost=field_class.boost, sortable=True)  # 中文分词器
            if field_class.document is True:  # 标记主内容字段（通常用于全文搜索的字段）
                content_field_name = field_class.index_fieldname
                schema_fields[field_class.index_fieldname].spelling = True  # 启用拼写建议

        # Fail more gracefully than relying on the backend to die if no fields，验证：确保至少添加了一个自定义字段
        # are found.
        if len(schema_fields) <= initial_key_count:
            raise SearchBackendError(
                "No fields were found in any search_indexes. Please correct this before attempting to search.")

        return (content_field_name, Schema(**schema_fields))# 返回主内容字段名和构建好的Schema对象

    def update(self, index, iterable, commit=True):  # 延迟初始化：如果尚未完成设置，先调用setup()方法
        if not self.setup_complete:
            self.setup()
# 刷新索引，确保获取最新的索引状态
# 这在多进程/多线程环境中很重要，避免写入冲突
        
        self.index = self.index.refresh() # 创建异步写入器，提供更好的写入性能
        writer = AsyncWriter(self.index)
    # 遍历所有要索引的对象
        for obj in iterable:
            try: # 准备索引文档：将Django对象转换为搜索索引文档
                doc = index.full_prepare(obj)
            except SkipDocument:  # 如果对象被标记为跳过索引（如某些条件不满足）
                self.log.debug(u"Indexing for object `%s` skipped", obj)
            else:
                # Really make sure it's unicode, because Whoosh won't have it any
                # other way.
                for key in doc:
                    doc[key] = self._from_python(doc[key])
# 确保所有字段值都是Unicode字符串
 # Whoosh严格要求Unicode格式，避免编码问题

                
                # Document boosts aren't supported in Whoosh 2.5.0+.
                if 'boost' in doc:  # 处理文档权重：Whoosh 2.5.0+ 不再支持文档级别的boost
                    del doc['boost']   # 移除boost字段，避免Whoosh报错

                try:   # 更新索引文档
                    writer.update_document(**doc)   # update_document方法：如果文档存在则更新，不存在则创建
                except Exception as e:   # 处理写入过程中的异常
                    if not self.silently_fail:  # 如果不允许静默失败，则重新抛出异常
                        raise

                    # We'll log the object identifier but won't include the actual object
                    # to avoid the possibility of that generating encoding errors while
                    # processing the log message:
                    # 记录错误日志，但不包含实际对象内容（避免编码错误）
                    self.log.error(
                        u"%s while preparing object for update" %
                        e.__class__.__name__,
                        exc_info=True,  # 包含完整的异常堆栈信息
                        extra={
                            "data": {
                                "index": index,  # 索引名称
                                "object": get_identifier(obj)}})   # 对象标识符（安全记录）

        if len(iterable) > 0:  # 提交更改：只有在有实际对象需要索引时才执行
         # 立即提交写入操作
        # 注意：这里无论commit参数如何都强制提交，因为Whoosh存在锁问题
            # For now, commit no matter what, as we run into locking issues
            # otherwise.
            writer.commit()  # 强制提交的原因：避免索引锁未释放导致后续操作阻塞

    def remove(self, obj_or_string, commit=True):
        if not self.setup_complete: # 延迟初始化：如果尚未完成设置，先调用setup()方法
            self.setup()

        self.index = self.index.refresh()   # 刷新索引，确保获取最新的索引状态
    # 获取对象的唯一标识符
    # 可以接受对象实例或直接传递标识符字符串
        whoosh_id = get_identifier(obj_or_string)

        try:
        # 通过查询删除指定文档
        # 构建查询：ID字段等于指定标识符的文档
            self.index.delete_by_query(
                q=self.parser.parse(
                    u'%s:"%s"' %  
                    (ID, whoosh_id)))  # 示例：_django_id:"blog.post.1"
        except Exception as e:  # 处理删除过程中的异常
            if not self.silently_fail:
                raise   # 不允许静默失败时重新抛出异常

            self.log.error(   # 记录错误日志，包含完整的异常信息
                "Failed to remove document '%s' from Whoosh: %s",
                whoosh_id,
                e,
                exc_info=True)   # 包含堆栈跟踪

    def clear(self, models=None, commit=True):
        if not self.setup_complete:
            self.setup()  # 刷新索引，确保获取最新的索引状态

        self.index = self.index.refresh()

        if models is not None:  # 验证models参数类型
            assert isinstance(models, (list, tuple))   # 必须是列表或元组

        try:
            if models is None:   # 清空整个索引：删除所有文档
                self.delete_index()
            else:  # 只清除指定模型的索引文档
                models_to_delete = []

                for model in models:   # 为每个模型构建删除查询条件
                    models_to_delete.append(
                        u"%s:%s" %
                        (DJANGO_CT, get_model_ct(model)))  # 构建模型标识符，格式：django_ct:"app_label.model_name"
            # 使用OR查询删除所有指定模型的文档
            # 示例：_django_ct:"blog.post" OR _django_ct:"auth.user"
                self.index.delete_by_query(
                    q=self.parser.parse(
                        u" OR ".join(models_to_delete)))
        except Exception as e:   # 处理清空过程中的异常
            if not self.silently_fail:
                raise

            if models is not None:   # 部分清空失败：记录具体哪些模型清除失败
                self.log.error(
                    "Failed to clear Whoosh index of models '%s': %s",
                    ','.join(models_to_delete),  # 记录所有目标模型
                    e,
                    exc_info=True)
            else:   # 完全清空失败：记录通用错误
                self.log.error(
                    "Failed to clear Whoosh index: %s", e, exc_info=True)

    def delete_index(self):
        # Per the Whoosh mailing list, if wiping out everything from the index,
        # it's much more efficient to simply delete the index files.
        if self.use_file_storage and os.path.exists(self.path):  # 文件存储模式：直接删除整个索引目录
            shutil.rmtree(self.path)  # 递归删除目录及其所有内容
        elif not self.use_file_storage:  # 内存存储模式：清理内存中的索引数据
            self.storage.clean()  # 清空RAM存储

        # Recreate everything.
        # 重新创建索引结构
        # 删除后需要重新初始化，以便后续的索引操作
        self.setup()

    def optimize(self):
        if not self.setup_complete:
            self.setup()

        self.index = self.index.refresh()  # 刷新索引确保获取最新状态
        self.index.optimize()  # 执行索引优化

    def calculate_page(self, start_offset=0, end_offset=None):
        # Prevent against Whoosh throwing an error. Requires an end_offset
        # greater than 0.
        if end_offset is not None and end_offset <= 0:  # 防止Whoosh抛出错误：end_offset必须大于0
            end_offset = 1

        # Determine the page.计算页码
        page_num = 0  

        if end_offset is None:  # 如果没有指定结束位置，使用一个很大的默认值
            end_offset = 1000000

        if start_offset is None:  # 如果没有指定起始位置，从0开始
            start_offset = 0

        page_length = end_offset - start_offset  # 计算每页长度

        if page_length and page_length > 0:  # 计算页码：起始位置 / 每页长度
            page_num = int(start_offset / page_length)

        # Increment because Whoosh uses 1-based page numbers.
        page_num += 1
        return page_num, page_length

    @log_query
# 查询日志装饰器，记录所有搜索查询的详细信息
# 这个装饰器会自动记录搜索查询的参数、执行时间、结果数量等信息
    
    def search(
            self,
            query_string,            # 搜索查询字符串
            sort_by=None,            # 排序字段
            start_offset=0,          # 起始偏移量
            end_offset=None,         # 结束偏移量
            fields='',               # 搜索字段限制
            highlight=False,         # 是否高亮结果
            facets=None,             # 分面搜索（Whoosh不支持）
            date_facets=None,        # 日期分面（Whoosh不支持）
            query_facets=None,       # 查询分面（Whoosh不支持）
            narrow_queries=None,     # 窄化查询条件
            spelling_query=None,     # 拼写检查查询
            within=None,             # 空间搜索范围内（Whoosh不支持）
            dwithin=None,            # 空间搜索距离内（Whoosh不支持）
            distance_point=None,     # 空间搜索基准点（Whoosh不支持）
            models=None,             # 限制搜索的模型
            limit_to_registered_models=None,    # 是否限制到注册模型
            result_class=None,                  # 结果类
            **kwargs):
        if not self.setup_complete:  # 延迟初始化
            self.setup()

        # A zero length query should return no results.空查询检查：零长度查询返回空结果
        if len(query_string) == 0:
            return {
                'results': [],
                'hits': 0,
            }

        query_string = force_str(query_string)  # 确保Unicode编码

        # A one-character query (non-wildcard) gets nabbed by a stopwords
        # filter and should yield zero results.
        # 单字符查询检查：非通配符的单字符查询返回空结果
        # 因为会被停用词过滤器捕获
        if len(query_string) <= 1 and query_string != u'*':
            return {
                'results': [],
                'hits': 0,
            }

        reverse = False  # 排序方向标志

        if sort_by is not None:  # 处理排序参数
            # Determine if we need to reverse the results and if Whoosh can
            # handle what it's being asked to sort by. Reversing is an
            # all-or-nothing action, unfortunately.
            sort_by_list = []
            reverse_counter = 0  # Whoosh要求所有排序字段使用相同的排序方向

            for order_by in sort_by: # 统计反向排序的字段数量
                if order_by.startswith('-'):  # 降序排序
                    reverse_counter += 1

            if reverse_counter and reverse_counter != len(sort_by):  # 验证排序方向一致性
                raise SearchBackendError("Whoosh requires all order_by fields"
                                         " to use the same sort direction")

            for order_by in sort_by:  # 构建排序字段列表并确定排序方向
                if order_by.startswith('-'):
                    sort_by_list.append(order_by[1:])  # 移除降序符号

                    if len(sort_by_list) == 1:
                        reverse = True  # 第一个字段决定整体排序方向
                else:
                    sort_by_list.append(order_by)

                    if len(sort_by_list) == 1:
                        reverse = False

            sort_by = sort_by_list[0]  # Whoosh只支持单字段排序

        if facets is not None:  # Whoosh不支持分面搜索，发出警告
            warnings.warn(
                "Whoosh does not handle faceting.",
                Warning,
                stacklevel=2)

        if date_facets is not None:
            warnings.warn(
                "Whoosh does not handle date faceting.",
                Warning,
                stacklevel=2)

        if query_facets is not None:
            warnings.warn(
                "Whoosh does not handle query faceting.",
                Warning,
                stacklevel=2)

        narrowed_results = None   # 窄化搜索结果
        self.index = self.index.refresh()   # 刷新索引

        if limit_to_registered_models is None:  # 确定是否限制到注册模型
            limit_to_registered_models = getattr(
                settings, 'HAYSTACK_LIMIT_TO_REGISTERED_MODELS', True)

        if models and len(models):  # 构建模型选择列表
            model_choices = sorted(get_model_ct(model) for model in models)
        elif limit_to_registered_models:   # 使用窄化查询，将结果限制到当前路由器处理的模型
            # Using narrow queries, limit the results to only models handled
            # with the current routers.
            model_choices = self.build_models_list()
        else:
            model_choices = []

        if len(model_choices) > 0:  # 添加模型限制到窄化查询
            if narrow_queries is None:
                narrow_queries = set()

            narrow_queries.add(' OR '.join(
                ['%s:%s' % (DJANGO_CT, rm) for rm in model_choices]))

        narrow_searcher = None

        if narrow_queries is not None:  # 处理窄化查询（用于结果过滤）
            # Potentially expensive? I don't see another way to do it in
            # Whoosh...
            narrow_searcher = self.index.searcher()

            for nq in narrow_queries:
                recent_narrowed_results = narrow_searcher.search(
                    self.parser.parse(force_str(nq)), limit=None)

                if len(recent_narrowed_results) <= 0:  # 如果窄化查询没有结果，直接返回空
                    return {
                        'results': [],
                        'hits': 0,
                    }

                if narrowed_results:  # 应用窄化过滤器
                    narrowed_results.filter(recent_narrowed_results)
                else:
                    narrowed_results = recent_narrowed_results

        self.index = self.index.refresh()

        if self.index.doc_count():  # 执行主搜索， 检查索引中是否有文档
            searcher = self.index.searcher()
            parsed_query = self.parser.parse(query_string)  # 解析查询字符串

            # In the event of an invalid/stopworded query, recover gracefully.
            if parsed_query is None:  # 处理无效/停用词查询
                return {
                    'results': [],
                    'hits': 0,
                }

            page_num, page_length = self.calculate_page( # 计算分页参数
                start_offset, end_offset)

            search_kwargs = {
                'pagelen': page_length,  # 每页长度
                'sortedby': sort_by,     # 排序字段
                'reverse': reverse,      # 排序方向
            }

            # Handle the case where the results have been narrowed.
            if narrowed_results is not None:  # 应用窄化过滤器
                search_kwargs['filter'] = narrowed_results

            try:  # 执行分页搜索
                raw_page = searcher.search_page(
                    parsed_query,
                    page_num,
                    **search_kwargs
                )
            except ValueError:  # 处理搜索异常
                if not self.silently_fail:
                    raise

                return {
                    'results': [],
                    'hits': 0,
                    'spelling_suggestion': None,
                }

            # Because as of Whoosh 2.5.1, it will return the wrong page of
            # results if you request something too high. :(
            if raw_page.pagenum < page_num:    # Whoosh 2.5.1 bug处理：请求过高页码时返回错误页面
                return {
                    'results': [],
                    'hits': 0,
                    'spelling_suggestion': None,
                }

            results = self._process_results(  # 处理搜索结果
                raw_page,
                highlight=highlight,
                query_string=query_string,
                spelling_query=spelling_query,
                result_class=result_class)
            searcher.close()

            if hasattr(narrow_searcher, 'close'):  # 关闭窄化搜索器
                narrow_searcher.close()

            return results
        else:  # 索引为空时的处理
            if self.include_spelling:
                if spelling_query:
                    spelling_suggestion = self.create_spelling_suggestion(
                        spelling_query)
                else:
                    spelling_suggestion = self.create_spelling_suggestion(
                        query_string)
            else:
                spelling_suggestion = None

            return {
                'results': [],
                'hits': 0,
                'spelling_suggestion': spelling_suggestion,
            }

    def more_like_this(
            self,
            model_instance,                    # 参考模型实例，基于此查找相似内容
            additional_query_string=None,      # 附加查询条件
            start_offset=0,                    # 起始偏移量
            end_offset=None,                   # 结束偏移量
            models=None,                       # 限制搜索的模型类型
            limit_to_registered_models=None,   # 是否限制到注册模型
            result_class=None,                 # 结果类
            **kwargs):
        if not self.setup_complete:
            self.setup()
        # 处理延迟加载模型：延迟模型的类名会不同（"RealClass_Deferred_fieldname"）
        # 使用具体模型类确保在注册表中能找到
        # Deferred models will have a different class ("RealClass_Deferred_fieldname")
        # which won't be in our registry:
        model_klass = model_instance._meta.concrete_model

        field_name = self.content_field_name # 主内容字段名，用于相似性比较
        narrow_queries = set()               # 窄化查询条件集合
        narrowed_results = None              # 窄化搜索结果
        self.index = self.index.refresh()    # 刷新索引

        if limit_to_registered_models is None:  # 确定是否限制到注册模型
            limit_to_registered_models = getattr(
                settings, 'HAYSTACK_LIMIT_TO_REGISTERED_MODELS', True)

        if models and len(models):  # 构建模型选择列表
            model_choices = sorted(get_model_ct(model) for model in models)
        elif limit_to_registered_models:   # 使用窄化查询，将结果限制到当前路由器处理的模型
            # Using narrow queries, limit the results to only models handled
            # with the current routers.
            model_choices = self.build_models_list()
        else:
            model_choices = []

        if len(model_choices) > 0:  # 添加模型限制到窄化查询
            if narrow_queries is None:
                narrow_queries = set()

            narrow_queries.add(' OR '.join(
                ['%s:%s' % (DJANGO_CT, rm) for rm in model_choices]))

        if additional_query_string and additional_query_string != '*':  # 添加附加查询条件
            narrow_queries.add(additional_query_string)

        narrow_searcher = None

        if narrow_queries is not None:   # 处理窄化查询（结果过滤）
            # Potentially expensive? I don't see another way to do it in
            # Whoosh...
            narrow_searcher = self.index.searcher()

            for nq in narrow_queries:
                recent_narrowed_results = narrow_searcher.search(
                    self.parser.parse(force_str(nq)), limit=None)

                if len(recent_narrowed_results) <= 0:  # 如果窄化查询没有结果，直接返回空
                    return {
                        'results': [],
                        'hits': 0,
                    }

                if narrowed_results:  # 应用窄化过滤器
                    narrowed_results.filter(recent_narrowed_results)
                else:
                    narrowed_results = recent_narrowed_results

        page_num, page_length = self.calculate_page(start_offset, end_offset)  # 计算分页参数

        self.index = self.index.refresh()
        raw_results = EmptyResults()  # 初始化空结果

        if self.index.doc_count():  # 检查索引中是否有文档
            query = "%s:%s" % (ID, get_identifier(model_instance))  # 构建查询：查找指定ID的文档
            searcher = self.index.searcher()
            parsed_query = self.parser.parse(query)
            results = searcher.search(parsed_query)

            if len(results):  # 如果找到参考文档，获取相似文档
                # 使用Whoosh内置的more_like_this功能
                # 基于主内容字段的文本相似性查找相关文档
                raw_results = results[0].more_like_this(
                    field_name, top=end_offset)  # top参数限制返回数量

            # Handle the case where the results have been narrowed. 应用窄化过滤器到相似结果
            if narrowed_results is not None and hasattr(raw_results, 'filter'):
                raw_results.filter(narrowed_results)

        try: # 处理分页结果
            raw_page = ResultsPage(raw_results, page_num, page_length)
        except ValueError:  # 处理分页异常
            if not self.silently_fail:
                raise

            return {
                'results': [],
                'hits': 0,
                'spelling_suggestion': None,
            }

        # Because as of Whoosh 2.5.1, it will return the wrong page of
        # results if you request something too high. :(
        if raw_page.pagenum < page_num:  # Whoosh 2.5.1 bug处理：请求过高页码时返回错误页面
            return {
                'results': [],
                'hits': 0,
                'spelling_suggestion': None,
            }

        results = self._process_results(raw_page, result_class=result_class)  # 处理最终结果
        searcher.close()

        if hasattr(narrow_searcher, 'close'):  # 关闭窄化搜索器
            narrow_searcher.close()

        return results

    def _process_results(
            self,
            raw_page,                        # Whoosh返回的原始分页结果
            highlight=False,                 # 是否高亮搜索结果
            query_string='',                 # 原始查询字符串，用于高亮
            spelling_query=None,             # 拼写检查查询
            result_class=None):              # 结果类，默认为SearchResult
        from haystack import connections
        results = []        # 处理后的结果列表
    # 重要：在切片之前先获取总命中数
    # 否则会导致分页计算错误
        # It's important to grab the hits first before slicing. Otherwise, this
        # can cause pagination failures.
        hits = len(raw_page)

        if result_class is None:  # 设置结果类，默认使用SearchResult
            result_class = SearchResult

        facets = {}  # 初始化返回数据结构
        spelling_suggestion = None  # 分面数据（Whoosh不支持，保持为空）
        unified_index = connections[self.connection_alias].get_unified_index()
        indexed_models = unified_index.get_indexed_models()  # 获取所有已索引的模型

        for doc_offset, raw_result in enumerate(raw_page):  # 遍历原始结果中的每个文档
            score = raw_page.score(doc_offset) or 0         # 获取文档得分，如果没有得分则设为0
            # 从结果中解析出Django模型的app_label和model_name
            # 格式：app_label.model_name
            app_label, model_name = raw_result[DJANGO_CT].split('.')
            additional_fields = {}  # 存储额外的字段数据
            model = haystack_get_model(app_label, model_name)  # 获取对应的Django模型类

            if model and model in indexed_models:  # 检查模型是否在索引中（有效的模型）
                for key, value in raw_result.items():   # 处理原始结果中的每个字段
                    index = unified_index.get_index(model)
                    string_key = str(key)   # 确保键为字符串

                    if string_key in index.fields and hasattr(  # 检查字段是否在索引中定义且有转换方法
                            index.fields[string_key], 'convert'):  # 检查字段是否在索引中定义且有转换方法
                        # Special-cased due to the nature of KEYWORD fields.
                        if index.fields[string_key].is_multivalued:
                            if value is None or len(value) == 0:
                                additional_fields[string_key] = []  # 空值设为空列表
                            else:
                                additional_fields[string_key] = value.split(
                                    ',')  # 将逗号分隔的字符串拆分为列表
                        else:
                            additional_fields[string_key] = index.fields[string_key].convert(
                                value)  # 使用字段定义的convert方法转换值
                    else:
                        additional_fields[string_key] = self._to_python(value)  # 普通字段，使用通用的Python类型转换

                del (additional_fields[DJANGO_CT])   # Django ContentType字段
                del (additional_fields[DJANGO_ID])     # Django对象ID字段

                if highlight:   # 处理搜索结果高亮
                    sa = StemmingAnalyzer()    # 词干分析器，用于分词
                    formatter = WhooshHtmlFormatter('em')    # HTML格式化器，使用<em>标签
                    terms = [token.text for token in sa(query_string)]    # 提取查询词

                    whoosh_result = whoosh_highlight(    # 执行高亮处理
                        additional_fields.get(self.content_field_name), # 要高亮的内容
                        terms,    # 需要高亮的查询词
                        sa,       # 分析器
                        ContextFragmenter(),    # 片段生成器
                        formatter               # HTML格式化器
                    )
                    additional_fields['highlighted'] = { # 将高亮结果添加到字段中
                        self.content_field_name: [whoosh_result],
                    }

                result = result_class(  # 创建SearchResult对象
                    app_label,        # 应用标签
                    model_name,        # 模型名称
                    raw_result[DJANGO_ID],    # 对象ID
                    score,                # 搜索得分
                    **additional_fields)    # 所有其他字段
                results.append(result)
            else:
                hits -= 1     # 如果模型无效，减少命中计数

        if self.include_spelling:  # 处理拼写建议
            if spelling_query:
                spelling_suggestion = self.create_spelling_suggestion(
                    spelling_query)
            else:
                spelling_suggestion = self.create_spelling_suggestion(
                    query_string)

        return { # 返回标准格式的搜索结果
            'results': results,  # 处理后的结果列表
            'hits': hits,        # 有效的命中总数
            'facets': facets,    # 分面数据（空字典）
            'spelling_suggestion': spelling_suggestion,    # 拼写建议
        }

    def create_spelling_suggestion(self, query_string):
        spelling_suggestion = None        # 初始化拼写建议
        reader = self.index.reader()      # 获取索引读取器
        corrector = reader.corrector(self.content_field_name)    # 创建拼写纠正器，基于主内容字段
        cleaned_query = force_str(query_string)                  # 确保查询字符串为Unicode

        if not query_string: # 空查询检查
            return spelling_suggestion

        # Clean the string.清理查询字符串：移除保留字
        for rev_word in self.RESERVED_WORDS:
            cleaned_query = cleaned_query.replace(rev_word, '')

        for rev_char in self.RESERVED_CHARACTERS:    # 清理查询字符串：移除特殊字符
            cleaned_query = cleaned_query.replace(rev_char, '')

        # Break it down.分词处理：将查询字符串拆分为单词列表
        query_words = cleaned_query.split()
        suggested_words = []    # 存储建议单词

        for word in query_words:    # 对每个单词进行拼写建议
            suggestions = corrector.suggest(word, limit=1)    # 获取拼写建议，限制返回1个最佳建议

            if len(suggestions) > 0:    # 如果有建议，使用第一个建议
                suggested_words.append(suggestions[0])     # 如果没有建议，保留原单词

        spelling_suggestion = ' '.join(suggested_words)   # 将建议单词重新组合成完整的查询字符串
        return spelling_suggestion

    def _from_python(self, value):
        """
        Converts Python values to a string for Whoosh.

        Code courtesy of pysolr.
        """
        # 处理日期时间对象
        if hasattr(value, 'strftime'):
            if not hasattr(value, 'hour'):    # 检查是否为日期对象（没有小时属性）
                value = datetime(value.year, value.month, value.day, 0, 0, 0)    # 将日期转换为完整的datetime对象（时间设为00:00:00）
        elif isinstance(value, bool):    # 处理布尔值
            if value:
                value = 'true'    # 布尔True转换为字符串'true'
            else:
                value = 'false'    # 布尔False转换为字符串'false'
        elif isinstance(value, (list, tuple)):    # 处理列表和元组
            value = u','.join([force_str(v) for v in value])    # 将多值字段转换为逗号分隔的字符串
        elif isinstance(value, (six.integer_types, float)):    # 处理数值类型（整数、浮点数）
            # Leave it alone.数值类型保持不变，Whoosh可以直接处理
            pass
        else:    # 处理其他类型（字符串等）
            value = force_str(value)    # 确保为Unicode字符串
        return value

    def _to_python(self, value):
        """
        Converts values from Whoosh to native Python values.

        A port of the same method in pysolr, as they deal with data the same way.
        """
        # 处理布尔值字符串
        if value == 'true':
            return True
        elif value == 'false':
            return False

        if value and isinstance(value, six.string_types):    # 处理字符串类型的值
            possible_datetime = DATETIME_REGEX.search(value)    # 使用正则表达式匹配日期时间格式

            if possible_datetime:
                date_values = possible_datetime.groupdict()    # 提取日期时间各个组成部分

                for dk, dv in date_values.items():    # 将字符串数字转换为整数
                    date_values[dk] = int(dv)

                return datetime(    # 构建Python datetime对象
                    date_values['year'],        # 年
                    date_values['month'],        # 月
                    date_values['day'],            # 日
                    date_values['hour'],            # 时
                    date_values['minute'],        # 分
                    date_values['second'])        # 秒

        try:    # 使用json加载值（处理列表、字典等序列化数据）
            # Attempt to use json to load the values.
            converted_value = json.loads(value)

            # Try to handle most built-in types.
            if isinstance(     # 处理大多数内置类型
                    converted_value,
                    (list,                 # 列表
                     tuple,                # 元组
                     set,                    # 集合
                     dict,                # 字典
                     six.integer_types,    # 整数类型
                     float,                # 浮点数
                     complex)):            # 复数
                return converted_value
        except BaseException:
            # If it fails (SyntaxError or its ilk) or we don't trust it,
            # continue on.
            # 如果JSON解析失败（语法错误等）或不信任结果，继续其他处理
            pass    

        return value    # 默认返回原值


class WhooshSearchQuery(BaseSearchQuery):
    def _convert_datetime(self, date):
        if hasattr(date, 'hour'):
            return force_str(date.strftime('%Y%m%d%H%M%S'))     # 完整的日期时间对象：格式化为YYYYMMDDHHMMSS
        else:
            return force_str(date.strftime('%Y%m%d000000'))    # 只有日期没有时间：时间部分补零为000000

    def clean(self, query_fragment):
        """
        Provides a mechanism for sanitizing user input before presenting the
        value to the backend.

        Whoosh 1.X differs here in that you can no longer use a backslash
        to escape reserved characters. Instead, the whole word should be
        quoted.
        """
        words = query_fragment.split()    # 将查询分割成单词
        cleaned_words = []                # 存储清理后的单词

        for word in words:    # 处理保留字：转换为小写
            if word in self.backend.RESERVED_WORDS:
                word = word.replace(word, word.lower())

            for char in self.backend.RESERVED_CHARACTERS: # 处理包含特殊字符的单词
                if char in word:
                    word = "'%s'" % word    # 用单引号引用整个单词
                    break                    # 找到一个特殊字符就足够，无需继续检查

            cleaned_words.append(word)

        return ' '.join(cleaned_words)        # 重新组合成完整的查询字符串

    def build_query_fragment(self, field, filter_type, value):
        from haystack import connections
        query_frag = ''    # 查询片段
        is_datetime = False    # 标记是否为日期时间类型

        if not hasattr(value, 'input_type_name'):    # 处理值类型：如果不是InputType，转换为适当的InputType
            # Handle when we've got a ``ValuesListQuerySet``...
            if hasattr(value, 'values_list'):    # 处理ValuesListQuerySet（Django查询集）
                value = list(value)    # 转换为列表

            if hasattr(value, 'strftime'):    # 检查是否为日期时间对象
                is_datetime = True

            if isinstance(value, six.string_types) and value != ' ':    # 处理字符串类型（非空字符串）
                # It's not an ``InputType``. Assume ``Clean``.不是InputType，假设使用Clean输入类型
                value = Clean(value)
            else:
                value = PythonData(value)    # 使用PythonData输入类型处理其他数据类型

        # Prepare the query using the InputType.
        prepared_value = value.prepare(self)    # 使用InputType准备查询值

        if not isinstance(prepared_value, (set, list, tuple)):    # 如果不是集合/列表/元组，转换为Whoosh兼容格式
            # Then convert whatever we get back to what pysolr wants if needed.
            prepared_value = self.backend._from_python(prepared_value)

        # 'content' is a special reserved word, much like 'pk' in
        # Django's ORM layer. It indicates 'no special field'.
        if field == 'content':    # 处理特殊字段'content'（表示无特定字段，全文搜索）
            index_fieldname = ''    # 空字段名表示全文搜索
        else:    # 获取字段的索引字段名
            index_fieldname = u'%s:' % connections[self._using].get_unified_index(
            ).get_index_fieldname(field)

        # 定义过滤类型到Whoosh语法的映射
        filter_types = {
            'content': '%s',        # 全文搜索
            'contains': '*%s*',     # 包含（前后通配符）
            'endswith': "*%s",      # 以...结尾（前缀通配符）
            'startswith': "%s*",    # 以...开头（后缀通配符）
            'exact': '%s',          # 精确匹配
            'gt': "{%s to}",        # 大于（开区间）
            'gte': "[%s to]",       # 大于等于（闭区间）
            'lt': "{to %s}",        # 小于（开区间）
            'lte': "[to %s]",       # 小于等于（闭区间）
            'fuzzy': u'%s~',        # 模糊匹配
        }

        if value.post_process is False:    # 如果值不需要后处理，直接使用准备值
            query_frag = prepared_value
        else:    # 处理文本搜索相关的过滤类型
            if filter_type in [
                'content',
                'contains',
                'startswith',
                'endswith',
                'fuzzy']:
                if value.input_type_name == 'exact':
                    query_frag = prepared_value
                else:
                    # Iterate over terms & incorportate the converted form of
                    # each into the query.
                    # 迭代术语并将每个术语的转换形式合并到查询中
                    terms = []

                    if isinstance(prepared_value, six.string_types):    # 分割字符串值或多值处理
                        possible_values = prepared_value.split(' ')     # 按空格分词
                    else:
                        if is_datetime is True:
                            prepared_value = self._convert_datetime(
                                prepared_value)    # 单值列表

                        possible_values = [prepared_value]

                    for possible_value in possible_values:    # 为每个可能的值构建查询片段
                        terms.append(
                            filter_types[filter_type] %
                            self.backend._from_python(possible_value))

                    if len(terms) == 1:    # 组合查询条件
                        query_frag = terms[0]
                    else:
                        query_frag = u"(%s)" % " AND ".join(terms)    # 多个条件用AND连接
            elif filter_type == 'in':    # 处理IN查询（多值匹配）
                in_options = []

                for possible_value in prepared_value:
                    is_datetime = False

                    if hasattr(possible_value, 'strftime'):
                        is_datetime = True

                    pv = self.backend._from_python(possible_value)

                    if is_datetime is True:
                        pv = self._convert_datetime(pv)

                    if isinstance(pv, six.string_types) and not is_datetime:    # 字符串值用引号包围，数值直接使用
                        in_options.append('"%s"' % pv)
                    else:
                        in_options.append('%s' % pv)

                query_frag = "(%s)" % " OR ".join(in_options)    # IN查询用OR连接
            elif filter_type == 'range':    # 处理范围查询
                start = self.backend._from_python(prepared_value[0])
                end = self.backend._from_python(prepared_value[1])

                if hasattr(prepared_value[0], 'strftime'):    # 处理日期时间范围
                    start = self._convert_datetime(start)

                if hasattr(prepared_value[1], 'strftime'):
                    end = self._convert_datetime(end)

                query_frag = u"[%s to %s]" % (start, end)    # Whoosh范围语法
            elif filter_type == 'exact':    # 处理精确匹配
                if value.input_type_name == 'exact':
                    query_frag = prepared_value
                else:
                    prepared_value = Exact(prepared_value).prepare(self)
                    query_frag = filter_types[filter_type] % prepared_value
            else:
                if is_datetime is True:    # 处理其他过滤类型
                    prepared_value = self._convert_datetime(prepared_value)

                query_frag = filter_types[filter_type] % prepared_value

        if len(query_frag) and not isinstance(value, Raw):    # 为查询片段添加括号（如果不是Raw类型）
            if not query_frag.startswith('(') and not query_frag.endswith(')'):
                query_frag = "(%s)" % query_frag

        return u"%s%s" % (index_fieldname, query_frag)    # 返回完整的查询片段（字段名 + 查询条件）

        # if not filter_type in ('in', 'range'):
        #     # 'in' is a bit of a special case, as we don't want to
        #     # convert a valid list/tuple to string. Defer handling it
        #     # until later...
        #     value = self.backend._from_python(value)


class WhooshEngine(BaseEngine):
    backend = WhooshSearchBackend    # 指定后端实现类
    query = WhooshSearchQuery        # 指定查询处理类
