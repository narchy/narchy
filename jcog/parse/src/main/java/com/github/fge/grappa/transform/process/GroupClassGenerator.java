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

package com.github.fge.grappa.transform.process;

import com.github.fge.grappa.run.context.Context;
import com.github.fge.grappa.run.context.ContextAware;
import com.github.fge.grappa.transform.CodeBlock;
import com.github.fge.grappa.transform.base.InstructionGraphNode;
import com.github.fge.grappa.transform.base.InstructionGroup;
import com.github.fge.grappa.transform.base.ParserClassNode;
import com.github.fge.grappa.transform.base.RuleMethod;
import me.qmx.jitescript.util.CodegenUtils;
import net.bytebuddy.dynamic.loading.ByteArrayClassLoader;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.FieldInsnNode;
import org.objectweb.asm.tree.InsnList;
import org.objectweb.asm.tree.InsnNode;
import org.objectweb.asm.tree.VarInsnNode;

import java.util.concurrent.ConcurrentHashMap;

import static org.objectweb.asm.Opcodes.*;

public abstract class GroupClassGenerator implements RuleMethodProcessor
{
    private final boolean forceCodeBuilding;
    protected ParserClassNode classNode;
    protected RuleMethod method;

    public static final ByteArrayClassLoader CL = new ByteArrayClassLoader(GroupClassGenerator.class.getClassLoader(), false, new ConcurrentHashMap<>());

    protected GroupClassGenerator(boolean forceCodeBuilding)
    {
        this.forceCodeBuilding = forceCodeBuilding;
    }

    @Override
    public final void process(ParserClassNode classNode, RuleMethod method)
    {
        this.classNode = classNode; //Objects.requireNonNull(classNode, "classNode");
        this.method = method; //Objects.requireNonNull(method, "method");

        var groups = method.getGroups();
        for (int i = 0, size = groups.size(); i < size; i++) {
            var group = groups.get(i);
            if (appliesTo(group.getRoot()))
                loadGroupClass(group);
        }
    }

    protected abstract boolean appliesTo(InstructionGraphNode group);

    private void loadGroupClass(InstructionGroup group)
    {
        createGroupClassType(group);
        var className = group.getGroupClassType().getClassName();

        Class groupClass = null;
        if (!forceCodeBuilding) {
            try {
                groupClass = CL.loadClass(className);
            } catch (ClassNotFoundException e) { }
        }

        if (groupClass == null) {
            var groupClassCode = generateGroupClassCode(group);
            group.setGroupClassCode(groupClassCode);
            try {

                CL.defineClass(className, groupClassCode);
            } catch (ClassNotFoundException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private void createGroupClassType(InstructionGroup group)
    {
        var s = classNode.name;
        /*
         * If the parser has no package, the group will be an embedded class
         * to the parser class
         */
        var lastSlash = classNode.name.lastIndexOf('/');
        var groupName = group.getName();
        var pkg = lastSlash >= 0 ? s.substring(0, lastSlash) : s;
        var groupClassInternalName = pkg  + '/' + groupName;
        group.setGroupClassType(Type.getObjectType(groupClassInternalName));
    }

    protected final byte[] generateGroupClassCode(InstructionGroup group)
    {
        var classWriter = new ClassWriter(ClassWriter.COMPUTE_FRAMES);
        generateClassBasics(group, classWriter);
        generateFields(group, classWriter);
        generateConstructor(classWriter);
        generateMethod(group, classWriter);
        return classWriter.toByteArray();
    }

    private void generateClassBasics(InstructionGroup group,
                                     ClassWriter cw)
    {
        cw.visit(V24, ACC_PUBLIC + ACC_FINAL + ACC_SYNTHETIC,
            group.getGroupClassType().getInternalName(), null,
            getBaseType().getInternalName(), null);
        cw.visitSource(classNode.sourceFile, null);
    }

    protected abstract Type getBaseType();

    private static void generateFields(InstructionGroup group,
                                       ClassWriter cw)
    {
        // TODO: fix the below comment; those "two members" should be split
        // CAUTION: the FieldNode has illegal access flags and an illegal
        // value field since these two members are reused for other
        // purposes, so we need to write out the field "manually" here
        // rather than just call "field.accept(cw)"
        for (var field: group.getFields())
            cw.visitField(ACC_PUBLIC + ACC_SYNTHETIC, field.name, field.desc,
                null, null);

    }

    private void generateConstructor(ClassWriter cw)
    {
        var mv = cw.visitMethod(ACC_PUBLIC, "<init>",
            CodegenUtils.sig(void.class, String.class), null, null);
        mv.visitVarInsn(ALOAD, 0);
        mv.visitVarInsn(ALOAD, 1);
        mv.visitMethodInsn(INVOKESPECIAL, getBaseType().getInternalName(),
            "<init>", CodegenUtils.sig(void.class, String.class), false);
        mv.visitInsn(RETURN);
        mv.visitMaxs(0, 0); // trigger automatic computing
    }

    protected abstract void generateMethod(InstructionGroup group,
        ClassWriter cw);

    protected static void insertSetContextCalls(InstructionGroup group, int localVarIx)
    {
        var instructions = group.getInstructions();
        var block = CodeBlock.newCodeBlock();

        for (var node: group.getNodes()) {
            if (!node.isCallOnContextAware())
                continue;

            var insn = node.getInstruction();

            if (node.getPredecessors().size() > 1) {
                // store the target of the call in a new local variable
                var loadTarget = node.getPredecessors()
                    .getFirst().getInstruction();

                block.clear().dup().astore(++localVarIx);
                instructions.insert(loadTarget, block.getInstructionList());

                // immediately before the call get the target from the local var
                // and set the context on it
                instructions.insertBefore(insn, new VarInsnNode(ALOAD, localVarIx));
            } else {
                // if we have only one predecessor the call does not take any
                // parameters and we can skip the storing and loading of the
                // invocation target
                instructions.insertBefore(insn, new InsnNode(DUP));
            }

            block.clear()
                .aload(1)
                .invokeinterface(CodegenUtils.p(ContextAware.class),
                    "setContext", CodegenUtils.sig(void.class, Context.class));

            instructions.insertBefore(insn, block.getInstructionList());
        }
    }

    protected static void convertXLoads(InstructionGroup group)
    {
        var owner = group.getGroupClassType().getInternalName();

        InsnList insnList;

        for (var node : group.getNodes()) {
            if (!node.isXLoad())
                continue;

            var insn = (VarInsnNode) node.getInstruction();
            var field = group.getFields().get(insn.var);
            var fieldNode = new FieldInsnNode(GETFIELD, owner,
                field.name, field.desc);

            insnList = group.getInstructions();

            // insert the correct GETFIELD after the xLoad
            insnList.insert(insn, fieldNode);
            // change the load to ALOAD 0
            insnList.set(insn, new VarInsnNode(ALOAD, 0));
        }
    }
}