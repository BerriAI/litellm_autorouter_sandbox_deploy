"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

import { AutorouterLearningsView } from "./_components/AutorouterLearningsView";

export default function AutorouterLearnings() {
  const { accessToken } = useAuthorized();

  return <AutorouterLearningsView accessToken={accessToken} />;
}
