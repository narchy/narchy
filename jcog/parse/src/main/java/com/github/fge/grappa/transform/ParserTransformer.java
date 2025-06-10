/*
 * Copyright (C) 2009-2011 Mathias Doenitz
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.fge.grappa.transform;

import com.github.fge.grappa.transform.base.ParserClassNode;
import com.github.fge.grappa.transform.generate.ActionClassGenerator;
import com.github.fge.grappa.transform.generate.ClassNodeInitializer;
import com.github.fge.grappa.transform.generate.ConstructorGenerator;
import com.github.fge.grappa.transform.generate.VarInitClassGenerator;
import com.github.fge.grappa.transform.process.*;
import com.google.common.annotations.VisibleForTesting;
import org.objectweb.asm.ClassWriter;

import java.io.IOException;
import java.util.List;

public enum ParserTransformer {
    ;

    private static final List<RuleMethodProcessor> ruleMethodProcessors = List.of(
            new UnusedLabelsRemover(),
            new ReturnInstructionUnifier(),
            new InstructionGraphCreator(),
            new ImplicitActionsConverter(),
            new InstructionGroupCreator(),
            new InstructionGroupPreparer(),
            new ActionClassGenerator(false),
            new VarInitClassGenerator(false),
            new RuleMethodRewriter(),
            new SuperCallRewriter(),
            new BodyWithSuperCallReplacer(),
            new VarFramingGenerator(),
            new LabellingGenerator(),
            new CachingGenerator()
    );

    // TODO: remove "synchronized" here
    // TODO: move elsewhere
    public static <T> Class<? extends T> transformParser(Class<T> parserClass) throws IOException {
        return (Class<? extends T>) extendParserClass(parserClass).getExtendedClass();
    }

    /**
     * Dump the bytecode of a transformed parser class
     *
     * <p>This method will run all bytecode transformations on the parser class
     * then return a dump of the bytecode as a byte array.</p>
     *
     * @param parserClass the parser class
     * @return a bytecode dump
     * @throws Exception FIXME
     * @see #extendParserClass(Class)
     */
    // TODO: poor exception specification
    public static byte[] getByteCode(Class<?> parserClass) throws IOException {
        return extendParserClass(parserClass).getClassCode();
    }

    @VisibleForTesting
    public static ParserClassNode extendParserClass(Class<?> parserClass) throws IOException {
        var n = new ParserClassNode(parserClass);
        new ClassNodeInitializer().process(n);
        runMethodTransformers(n);
        new ConstructorGenerator().process(n);
        defineExtendedParserClass(n);
        return n;
    }

    // TODO: poor exception handling again
    private static void runMethodTransformers(ParserClassNode classNode) {

        // TODO: comment above may be right, but it's still dangerous
        // iterate through all rule methods
        // since the ruleMethods map on the classnode is a treemap we get the
        // methods sorted by name which puts all super methods first (since they
        // are prefixed with one or more '$')

        classNode.getRuleMethods().values().forEach(m -> {
            if (!m.hasDontExtend()) {
                for (var methodProcessor : ruleMethodProcessors)
                    if (methodProcessor.appliesTo(classNode, m)) {
                        try {
                            methodProcessor.process(classNode, m);
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
            }
            if (!m.isGenerationSkipped())
                classNode.methods.add(m);
        });

    }

    private static void defineExtendedParserClass(ParserClassNode node) {
        var writer = new ClassWriter(ClassWriter.COMPUTE_FRAMES);

        node.accept(writer);
        node.setClassCode(writer.toByteArray());

        var className = node.name.replace('/', '.');
        var bytecode = node.getClassCode();

        try {
            node.setExtendedClass(
                    GroupClassGenerator.CL.defineClass(className, bytecode));
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }
}