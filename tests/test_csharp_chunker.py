"""Tests for C# Tree-sitter chunker — class/method boundary detection."""

from pipeline.csharp_chunker import chunk_csharp


SIMPLE_CLASS = """\
namespace Ping.Shipping
{
    public class ShipmentValidator
    {
        public bool Validate(string id)
        {
            return !string.IsNullOrEmpty(id);
        }
    }
}
"""

INTERFACE = """\
namespace Ping.Shipping
{
    public interface IShipmentService
    {
        Task<ShipResult> Ship(ShipRequest request);
        void Cancel(string id);
    }
}
"""

CLASS_WITH_INTERFACE = """\
namespace Ping.Shipping.Services
{
    public class ShipmentService : IShipmentService, IDisposable
    {
        private readonly ILogger _logger;

        public ShipmentService(ILogger logger)
        {
            _logger = logger;
        }

        public Task<ShipResult> Ship(ShipRequest request)
        {
            return Task.FromResult(new ShipResult());
        }

        public void Cancel(string id)
        {
            _logger.Log("Cancelled");
        }

        public void Dispose()
        {
        }
    }
}
"""

LARGE_CLASS = """\
namespace Ping.Big
{
    public class BigService : IBigService
    {
        private readonly IRepo _repo;

        public BigService(IRepo repo)
        {
            _repo = repo;
        }

        public void MethodOne()
        {
            // """ + "x" * 2000 + """
        }

        public void MethodTwo()
        {
            // """ + "y" * 2000 + """
        }
    }
}
"""

FILE_SCOPED_NAMESPACE = """\
namespace Ping.Modern;

public class ModernClass
{
    public void DoStuff() { }
}
"""

ENUM = """\
namespace Ping.Enums
{
    public enum Status
    {
        Active,
        Inactive,
        Pending
    }
}
"""

NO_CLASSES = """\
using System;
using System.Linq;
// Just some using statements
"""


class TestSimpleClass:
    def test_single_chunk(self):
        chunks = chunk_csharp(SIMPLE_CLASS)
        assert len(chunks) == 1
        assert chunks[0].class_name == "ShipmentValidator"
        assert chunks[0].namespace == "Ping.Shipping"
        assert chunks[0].is_interface is False

    def test_context_header(self):
        chunks = chunk_csharp(SIMPLE_CLASS)
        assert "// Namespace: Ping.Shipping" in chunks[0].text
        assert "// Class: ShipmentValidator" in chunks[0].text


class TestInterface:
    def test_parsed_as_interface(self):
        chunks = chunk_csharp(INTERFACE)
        assert len(chunks) == 1
        assert chunks[0].is_interface is True
        assert chunks[0].class_name == "IShipmentService"

    def test_interface_header(self):
        chunks = chunk_csharp(INTERFACE)
        assert "// Interface: IShipmentService" in chunks[0].text


class TestClassWithInterface:
    def test_detects_implements(self):
        chunks = chunk_csharp(CLASS_WITH_INTERFACE)
        # Small class, should be one chunk
        assert len(chunks) == 1
        assert "IShipmentService" in chunks[0].implements_interfaces

    def test_base_types_in_header(self):
        chunks = chunk_csharp(CLASS_WITH_INTERFACE)
        assert "IShipmentService" in chunks[0].text


class TestLargeClass:
    def test_splits_by_method(self):
        chunks = chunk_csharp(LARGE_CLASS, chunk_size=512)
        # Should have constructor + 2 methods = 3 chunks
        assert len(chunks) == 3

    def test_method_chunks_have_context(self):
        chunks = chunk_csharp(LARGE_CLASS, chunk_size=512)
        for chunk in chunks:
            assert "// Namespace: Ping.Big" in chunk.text
            assert "// Class: BigService" in chunk.text

    def test_method_chunks_include_constructor(self):
        chunks = chunk_csharp(LARGE_CLASS, chunk_size=512)
        # Non-constructor method chunks should include constructor text
        method_chunks = [c for c in chunks if c.method_name not in ("", "constructor")]
        for chunk in method_chunks:
            assert "BigService(IRepo repo)" in chunk.text

    def test_method_names_extracted(self):
        chunks = chunk_csharp(LARGE_CLASS, chunk_size=512)
        names = [c.method_name for c in chunks]
        assert "constructor" in names
        assert "MethodOne" in names
        assert "MethodTwo" in names

    def test_implements_on_all_chunks(self):
        chunks = chunk_csharp(LARGE_CLASS, chunk_size=512)
        for chunk in chunks:
            assert "IBigService" in chunk.implements_interfaces


class TestFileScopedNamespace:
    def test_detects_namespace(self):
        chunks = chunk_csharp(FILE_SCOPED_NAMESPACE)
        assert len(chunks) == 1
        assert chunks[0].namespace == "Ping.Modern"


class TestEnum:
    def test_enum_as_single_chunk(self):
        chunks = chunk_csharp(ENUM)
        assert len(chunks) == 1
        assert chunks[0].class_name == "Status"


class TestNoClasses:
    def test_whole_file_fallback(self):
        chunks = chunk_csharp(NO_CLASSES)
        assert len(chunks) == 1
        assert chunks[0].class_name == ""


class TestChunkIndices:
    def test_sequential_indices(self):
        chunks = chunk_csharp(LARGE_CLASS, chunk_size=512)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))
