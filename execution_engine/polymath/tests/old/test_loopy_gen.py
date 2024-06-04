# from polymath.pmlang.antlr_generator.parser import FileStream, CommonTokenStream, PMLangParser, ParseTreeWalker
# from polymath.pmlang.antlr_generator.lexer import PMLangLexer
# from polymath.pmlang.symbols import PMLangListener
# import loopy as lp
#
# import os
#
#
# def test_linear():
#     file = "linear.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#
#     input_file = FileStream(full_path)
#     lexer = PMLangLexer(input_file)
#     stream = CommonTokenStream(lexer)
#     parser = PMLangParser(stream)
#     tree = parser.pmlang()
#     output_dir, output_file = os.path.split(file)
#     output_mgdfg = f"{base_path}/outputs/{output_file[:-3]}.pb"
#     pmlang_graph = PMLangListener(full_path)
#     walker = ParseTreeWalker()
#     walker.walk(pmlang_graph, tree)
#     ext_full_path = os.path.abspath(base_path) + "/outputs"
#     lin_reg_comp = pmlang_graph.components["linear_regression"]
#
#     for expr in lin_reg_comp.expressions:
#         if expr.expr_type == "assignment":
#             lp.make_function(expr.assignee_domain, expr.loopy_stmt, "...", assumptions=expr.assumption_domain)
#
# def test_yolo():
#     file = "yolodnn.pm"
#     base_path = f"./pmlang_examples"
#     full_path = f"./pmlang_examples/{file}"
#     input_file = FileStream(full_path)
#     lexer = PMLangLexer(input_file)
#     stream = CommonTokenStream(lexer)
#     parser = PMLangParser(stream)
#     tree = parser.pmlang()
#     output_dir, output_file = os.path.split(file)
#     output_mgdfg = f"{base_path}/outputs/{output_file[:-3]}.pb"
#     pmlang_graph = PMLangListener(full_path)
#     walker = ParseTreeWalker()
#     walker.walk(pmlang_graph, tree)
#
#     for comp_name, comp in pmlang_graph.components.items():
#         if comp_name == "main":
#             continue
#         for expr in comp.expressions:
#             if expr.expr_type == "assignment":
#                 lp.make_function(expr.assignee_domain, expr.loopy_stmt, "...", assumptions=expr.assumption_domain)
#
# # def test_backprop():
# #     file = "backpropagation.pm"
# #     base_path = f"./pmlang_examples"
# #     full_path = f"./pmlang_examples/{file}"
# #     input_file = FileStream(full_path)
# #     lexer = PMLangLexer(input_file)
# #     stream = CommonTokenStream(lexer)
# #     parser = PMLangParser(stream)
# #     tree = parser.pmlang()
# #     output_dir, output_file = os.path.split(file)
# #     output_mgdfg = f"{base_path}/outputs/{output_file[:-3]}.pb"
# #     pmlang_graph = PMLangListener(full_path)
# #     walker = ParseTreeWalker()
# #     walker.walk(pmlang_graph, tree)
# #     ext_full_path = os.path.abspath(base_path) + "/outputs"
#
#
#
#
#
#
#
# # def test_logistic():
# #     file = "logistic.pm"
# #     base_path = f"./pmlang_examples"
# #     full_path = f"./pmlang_examples/{file}"
# #     input_file = FileStream(full_path)
# #     lexer = PMLangLexer(input_file)
# #     stream = CommonTokenStream(lexer)
# #     parser = PMLangParser(stream)
# #     tree = parser.pmlang()
# #     output_dir, output_file = os.path.split(file)
# #     output_mgdfg = f"{base_path}/outputs/{output_file[:-3]}.pb"
# #     pmlang_graph = PMLangListener(full_path)
# #     walker = ParseTreeWalker()
# #     walker.walk(pmlang_graph, tree)
# #     # load_store.save_program(pmlang_graph.program, output_mgdfg)
# #
# # def test_recommender():
# #     file = "recommender.pm"
# #     base_path = f"./pmlang_examples"
# #     full_path = f"./pmlang_examples/{file}"
# #     input_file = FileStream(full_path)
# #     lexer = PMLangLexer(input_file)
# #     stream = CommonTokenStream(lexer)
# #     parser = PMLangParser(stream)
# #     tree = parser.pmlang()
# #     output_dir, output_file = os.path.split(file)
# #     output_mgdfg = f"{base_path}/outputs/{output_file[:-3]}.pb"
# #     pmlang_graph = PMLangListener(full_path)
# #     walker = ParseTreeWalker()
# #     walker.walk(pmlang_graph, tree)
# #     ext_full_path = os.path.abspath(base_path) + "/outputs"
# #     visualize_component(pmlang_graph.components["rec_model"], ext_full_path)
# #     # load_store.save_program(pmlang_graph.program, output_mgdfg)
# #
# # def test_lenet():
# #     file = "lenet.pm"
# #     base_path = f"./pmlang_examples"
# #     full_path = f"./pmlang_examples/{file}"
# #     input_file = FileStream(full_path)
# #     lexer = PMLangLexer(input_file)
# #     stream = CommonTokenStream(lexer)
# #     parser = PMLangParser(stream)
# #     tree = parser.pmlang()
# #     output_dir, output_file = os.path.split(file)
# #     output_mgdfg = f"{base_path}/outputs/{output_file[:-3]}.pb"
# #     pmlang_graph = PMLangListener(full_path)
# #     walker = ParseTreeWalker()
# #     walker.walk(pmlang_graph, tree)
# #     ext_full_path = os.path.abspath(base_path) + "/outputs"
# #     visualize_component(pmlang_graph.components["main"], ext_full_path)
# #     # load_store.save_program(pmlang_graph.program, output_mgdfg)
# #
# # def test_yolo():
# #     file = "yolodnn.pm"
# #     base_path = f"./pmlang_examples"
# #     full_path = f"./pmlang_examples/{file}"
# #     input_file = FileStream(full_path)
# #     lexer = PMLangLexer(input_file)
# #     stream = CommonTokenStream(lexer)
# #     parser = PMLangParser(stream)
# #     tree = parser.pmlang()
# #     output_dir, output_file = os.path.split(file)
# #     output_mgdfg = f"{base_path}/outputs/{output_file[:-3]}.pb"
# #     pmlang_graph = PMLangListener(full_path)
# #     walker = ParseTreeWalker()
# #     walker.walk(pmlang_graph, tree)
# #
# # def test_resnet():
# #     file = "resnet18.pm"
# #     base_path = f"./pmlang_examples"
# #     full_path = f"./pmlang_examples/{file}"
# #     input_file = FileStream(full_path)
# #     lexer = PMLangLexer(input_file)
# #     stream = CommonTokenStream(lexer)
# #     parser = PMLangParser(stream)
# #     tree = parser.pmlang()
# #     output_dir, output_file = os.path.split(file)
# #     output_mgdfg = f"{base_path}/outputs/{output_file[:-3]}.pb"
# #     pmlang_graph = PMLangListener(full_path)
# #     walker = ParseTreeWalker()
# #     walker.walk(pmlang_graph, tree)
