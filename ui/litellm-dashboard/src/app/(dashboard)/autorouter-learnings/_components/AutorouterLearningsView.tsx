"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw, Trash2 } from "lucide-react";
import React, { useState } from "react";

import {
  AutorouterLearning,
  clearAutorouterLearnings,
  fetchAutorouterLearnings,
} from "@/components/networking";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";

const LEARNINGS_KEY = "autorouterLearnings" as const;

interface AutorouterLearningsViewProps {
  accessToken: string | null;
}

function learningTimestamp(name: string): string {
  const match = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/.exec(name);
  if (!match) return name;
  const [, year, month, day, hour, minute, second] = match;
  const parsed = new Date(`${year}-${month}-${day}T${hour}:${minute}:${second}Z`);
  return Number.isNaN(parsed.getTime()) ? name : parsed.toLocaleString();
}

const LearningCard: React.FC<{ learning: AutorouterLearning }> = ({ learning }) => (
  <div className="rounded-lg border border-gray-200 bg-white">
    <div className="border-b border-gray-100 px-4 py-2 text-xs text-gray-500">
      {learningTimestamp(learning.name)}
    </div>
    <pre className="max-h-96 overflow-auto whitespace-pre-wrap px-4 py-3 text-sm text-gray-800">
      {learning.content}
    </pre>
  </div>
);

export const AutorouterLearningsView: React.FC<AutorouterLearningsViewProps> = ({ accessToken }) => {
  const [isClearOpen, setIsClearOpen] = useState(false);
  const queryClient = useQueryClient();

  const { data, isLoading, isFetching, error } = useQuery({
    queryKey: [LEARNINGS_KEY],
    queryFn: () => {
      if (!accessToken) throw new Error("Access token required");
      return fetchAutorouterLearnings(accessToken);
    },
    enabled: !!accessToken,
  });

  const clearMutation = useMutation({
    mutationFn: () => {
      if (!accessToken) throw new Error("Access token required");
      return clearAutorouterLearnings(accessToken);
    },
    onSuccess: (result) => {
      toast.success(`Cleared ${result.deleted} learning${result.deleted === 1 ? "" : "s"}`);
      setIsClearOpen(false);
      void queryClient.invalidateQueries({ queryKey: [LEARNINGS_KEY] });
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const learnings = data?.learnings ?? [];

  return (
    <div className="space-y-4 p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-gray-900">Autorouter Learnings</h1>
          <p className="mt-1 text-sm text-gray-500">
            What the learning router has stored from strong-tier responses, newest first. Injected into
            matching later asks so a cheaper model can handle them.
          </p>
        </div>
        <div className="flex shrink-0 gap-2">
          <Button
            variant="secondary"
            onClick={() => void queryClient.invalidateQueries({ queryKey: [LEARNINGS_KEY] })}
            disabled={isFetching}
          >
            <RefreshCw className={`mr-2 h-4 w-4 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button variant="destructive" onClick={() => setIsClearOpen(true)} disabled={learnings.length === 0}>
            <Trash2 className="mr-2 h-4 w-4" />
            Clear all
          </Button>
        </div>
      </div>

      {data ? (
        <p className="text-sm text-gray-500">
          {data.count} stored in <code className="text-xs">{data.store_dir}</code>
        </p>
      ) : null}

      {error ? (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          Could not load learnings: {(error as Error).message}
        </div>
      ) : null}

      {isLoading ? <p className="text-sm text-gray-500">Loading...</p> : null}

      {!isLoading && !error && learnings.length === 0 ? (
        <div className="rounded-lg border border-gray-200 bg-gray-50 p-8 text-center">
          <p className="text-sm text-gray-600">No learnings stored yet.</p>
          <p className="mt-1 text-sm text-gray-500">
            Send requests to <code className="text-xs">moe-learning-router</code>. Responses that route to
            the COMPLEX or REASONING tier get stored here.
          </p>
        </div>
      ) : null}

      <div className="space-y-3">
        {learnings.map((learning) => (
          <LearningCard key={learning.name} learning={learning} />
        ))}
      </div>

      <DeleteResourceModal
        isOpen={isClearOpen}
        title="Clear all learnings"
        message={`This permanently deletes all ${learnings.length} stored learnings. The router will start over with an empty store.`}
        onCancel={() => setIsClearOpen(false)}
        onOk={() => clearMutation.mutate()}
        confirmLoading={clearMutation.isPending}
      />
    </div>
  );
};
